import UIKit
import AVFoundation
import AudioToolbox
import Network

class ViewController: UIViewController {
    
    // MARK: - Properties
    private var captureDevice: AVCaptureDevice!
    private var captureSession: AVCaptureSession!
    private var photoOutput: AVCapturePhotoOutput!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var httpServer: HTTPServer!
    
    private var pendingCompletion: ((HTTPResponse) -> Void)?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        print("=== App Starting ===")
        setupCamera()
        setupHTTPServer()
        displayServerInfo()
    }
    
    // MARK: - setupCamera
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        var backCamera: AVCaptureDevice?
        
        // ✨ iPhone 14 Pro: Try to get the MAIN 48MP camera specifically
        if #available(iOS 15.4, *) {
            // Prefer the triple camera (includes 48MP main)
            if let tripleCamera = AVCaptureDevice.default(.builtInTripleCamera, for: .video, position: .back) {
                backCamera = tripleCamera
                captureDevice = tripleCamera
                print("✅   Using Triple Camera System (48MP Main + Ultra Wide + Telephoto)")
                print("     Constituent devices: \(tripleCamera.constituentDevices.count)")
            }
        }
        
        // Fallback: Wide angle only
        if backCamera == nil,
           let wideCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) {
            backCamera = wideCamera
            captureDevice = wideCamera
            print("⚠️   Using Single Wide Angle Camera (calibration data unlikely)")
        }
        
        guard let camera = backCamera else {
            print("❌   No back camera available")
            return
        }
        
        do {
            // Disable geometric distortion correction
            try camera.lockForConfiguration()
            if camera.isGeometricDistortionCorrectionSupported {
                camera.isGeometricDistortionCorrectionEnabled = false
                print("✅   Device: Geometric distortion correction DISABLED")
            }
            camera.unlockForConfiguration()
            
            let input = try AVCaptureDeviceInput(device: camera)
            photoOutput = AVCapturePhotoOutput()
            
            captureSession.beginConfiguration()
            
            if captureSession.canAddInput(input) && captureSession.canAddOutput(photoOutput) {
                captureSession.addInput(input)
                captureSession.addOutput(photoOutput)
                
                // Configure for calibration data
                photoOutput.isContentAwareDistortionCorrectionEnabled = false
                print("✅   PhotoOutput: Content-aware distortion correction DISABLED")
                
                // Enable multi-camera delivery
                #if !os(macOS)
                if #available(iOS 13, *) {
                    if photoOutput.isVirtualDeviceConstituentPhotoDeliverySupported {
                        photoOutput.isVirtualDeviceConstituentPhotoDeliveryEnabled = true
                        print("✅   PhotoOutput: Virtual device constituent delivery ENABLED")
                    }
                }
                #endif
                
                // Check calibration support
                if photoOutput.isCameraCalibrationDataDeliverySupported {
                    print("✅   Camera calibration data delivery IS SUPPORTED")
                } else {
                    print("⚠️   Camera calibration data delivery NOT SUPPORTED")
                }
                
                // Enable depth if available
                if photoOutput.isDepthDataDeliverySupported {
                    photoOutput.isDepthDataDeliveryEnabled = true
                    print("✅   Depth data delivery enabled")
                }
                
                // ✨ Enable max photo quality for 48MP
                if #available(iOS 13, *) {
                    photoOutput.maxPhotoQualityPrioritization = .quality
                    print("✅   Photo quality set to max")
                }
                
                setupLivePreview()
            }
            
            captureSession.commitConfiguration()
            
        } catch {
            print("❌ Error setting up camera: \(error)")
        }
    }
    
    private func setupLivePreview() {
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.layer.bounds
        view.layer.addSublayer(previewLayer)
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.layer.bounds
    }
    
    
    // MARK: - HTTP Server Setup
    private func setupHTTPServer() {
        print("🚀 Setting up HTTP server...")
        httpServer = HTTPServer()
        
        httpServer.addHandler(path: "/capture", method: "POST") { [weak self] request, completion in
            print("📸 Capture request received")
            self?.handleCaptureRequest(completion: completion)
        }
        
        httpServer.addHandler(path: "/health", method: "GET") { _, completion in
            print("❤️ Health check request received")
            let healthData: [String: Any] = ["status": "ok", "timestamp": Date().timeIntervalSince1970]
            do {
                let jsonData = try JSONSerialization.data(withJSONObject: healthData)
                completion(HTTPResponse(statusCode: 200, data: jsonData, contentType: "application/json"))
            } catch {
                let errorData = "Internal Server Error".data(using: .utf8) ?? Data()
                completion(HTTPResponse(statusCode: 500, data: errorData, contentType: "text/plain"))
            }
        }
        
        httpServer.start(port: 8080)
        print("✅ HTTP Server setup complete")
    }
    
    private func handleCaptureRequest(completion: @escaping (HTTPResponse) -> Void) {
        print("🎯 Storing completion handler and triggering capture")
        
        if pendingCompletion != nil {
            print("⚠️ Capture already in progress")
            let errorData = "Capture already in progress".data(using: .utf8) ?? Data()
            completion(HTTPResponse(statusCode: 409, data: errorData, contentType: "text/plain"))
            return
        }
        
        self.pendingCompletion = completion
        self.capturePhoto()
    }

    private func capturePhoto() {
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        
        // ✨ Request MAXIMUM resolution (48MP on iPhone 14 Pro)
        if #available(iOS 16.0, *) {
            settings.maxPhotoDimensions = photoOutput.maxPhotoDimensions
            settings.photoQualityPrioritization = .quality
            print("📸 Requesting max dimensions: \(photoOutput.maxPhotoDimensions)")
        } else {
            settings.isHighResolutionPhotoEnabled = true
        }
        
        // Enable constituent photo delivery from ALL cameras
        #if !os(macOS)
        if #available(iOS 13, *) {
            if photoOutput.isVirtualDeviceConstituentPhotoDeliverySupported,
               let device = captureDevice {
                settings.virtualDeviceConstituentPhotoDeliveryEnabledDevices = device.constituentDevices
                print("✅ Enabled delivery from \(device.constituentDevices.count) constituent cameras:")
                for (index, constituentDevice) in device.constituentDevices.enumerated() {
                    print("   [\(index)] \(constituentDevice.localizedName) - \(constituentDevice.deviceType)")
                }
            }
        }
        #endif
        
        // Enable calibration data
        if photoOutput.isCameraCalibrationDataDeliverySupported {
            settings.isCameraCalibrationDataDeliveryEnabled = true
            print("✅ Capture: Calibration data delivery ENABLED")
        } else {
            print("⚠️ Capture: Calibration data NOT available")
        }
        
        // Enable depth
        if photoOutput.isDepthDataDeliverySupported {
            settings.isDepthDataDeliveryEnabled = true
        }
        AudioServicesPlaySystemSound(1108)
        
        print("📸 Capturing photo with all quality settings...")
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    private func displayServerInfo() {
        if let ip = getLocalIPAddress() {
            print("📱 iPhone IP Address: \(ip)")
            print("🌐 Server URL: http://\(ip):8080")
            
            DispatchQueue.main.async {
                let alert = UIAlertController(title: "Server Started",
                                            message: "Connect to:\nhttp://\(ip):8080\n\nEndpoints:\n• POST /capture\n• GET /health",
                                            preferredStyle: .alert)
                alert.addAction(UIAlertAction(title: "OK", style: .default))
                self.present(alert, animated: true)
            }
        } else {
            print("❌ Could not get IP address")
        }
    }
    
    private func getLocalIPAddress() -> String? {
        var address: String?
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        
        guard getifaddrs(&ifaddr) == 0 else { return nil }
        guard let firstAddr = ifaddr else { return nil }
        
        for ifptr in sequence(first: firstAddr, next: { $0.pointee.ifa_next }) {
            let interface = ifptr.pointee
            let addrFamily = interface.ifa_addr.pointee.sa_family
            
            if addrFamily == UInt8(AF_INET) {
                let name = String(cString: interface.ifa_name)
                if name == "en0" || name == "bridge100" || name.hasPrefix("en") {
                    var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    getnameinfo(interface.ifa_addr, socklen_t(interface.ifa_addr.pointee.sa_len),
                               &hostname, socklen_t(hostname.count),
                               nil, socklen_t(0), NI_NUMERICHOST)
                    let addr = String(cString: hostname)
                    if !addr.hasPrefix("127.") && !addr.isEmpty {
                        address = addr
                    }
                }
            }
        }
        freeifaddrs(ifaddr)
        return address
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        print("📸 Photo capture callback triggered")
        
        guard let completion = pendingCompletion else {
            print("⚠️ No pending completion handler!")
            return
        }
        
        defer {
            pendingCompletion = nil
            print("✅ Cleared pending completion handler")
        }
        
        if let error = error {
            print("❌ Error capturing photo: \(error.localizedDescription)")
            let errorData = "Error capturing photo: \(error.localizedDescription)".data(using: .utf8) ?? Data()
            completion(HTTPResponse(statusCode: 500, data: errorData, contentType: "text/plain"))
            return
        }
        
        guard let imageData = photo.fileDataRepresentation() else {
            print("❌ Could not get image data")
            let errorData = "Could not get image data".data(using: .utf8) ?? Data()
            completion(HTTPResponse(statusCode: 500, data: errorData, contentType: "text/plain"))
            return
        }
        
        print("✅ Image data captured: \(imageData.count) bytes")
        
        // Check if we have native calibration data
        if let calibrationData = photo.cameraCalibrationData {
            print("🎉 NATIVE camera calibration data available!")
            sendPhotoWithCalibration(imageData: imageData, photo: photo, calibrationData: calibrationData, completion: completion)
        } else {
            print("⚠️ No native calibration data - using fallback estimation")
            sendPhotoWithFallbackCalibration(imageData: imageData, photo: photo, completion: completion)
        }
    }
    
    private func sendPhotoWithCalibration(imageData: Data, photo: AVCapturePhoto, calibrationData: AVCameraCalibrationData, completion: @escaping (HTTPResponse) -> Void) {
        let intrinsics = calibrationData.intrinsicMatrix
        let dimensions = calibrationData.intrinsicMatrixReferenceDimensions
        let extrinsics = calibrationData.extrinsicMatrix
        
        // Convert intrinsic matrix to standard K format
        let K = [
            [Double(intrinsics.columns.0.x), Double(intrinsics.columns.1.x), Double(intrinsics.columns.2.x)],  // [fx, 0, 0]
            [Double(intrinsics.columns.0.y), Double(intrinsics.columns.1.y), Double(intrinsics.columns.2.y)],  // [0, fy, 0]§
            [Double(intrinsics.columns.0.z), Double(intrinsics.columns.1.z), Double(intrinsics.columns.2.z)]   // [cx, cy, 1]
        ]
        
        print(intrinsics)
        
        // Print to verify
        print("📐 Intrinsics Matrix (K):")
        print("   [[fx=\(K[0][0]), 0=\(K[0][1]), cx=\(K[0][2])],")
        print("    [0=\(K[1][0]), fy=\(K[1][1]), cy=\(K[1][2])],")
        print("    [0=\(K[2][0]), 0=\(K[2][1]), 1=\(K[2][2])]]")
        
        // Extract extrinsic matrix (4x3 -> rotation + translation)
        let R = [
            [Double(extrinsics.columns.0.x), Double(extrinsics.columns.1.x), Double(extrinsics.columns.2.x)],
            [Double(extrinsics.columns.0.y), Double(extrinsics.columns.1.y), Double(extrinsics.columns.2.y)],
            [Double(extrinsics.columns.0.z), Double(extrinsics.columns.1.z), Double(extrinsics.columns.2.z)]
        ]
        let t = [
            Double(extrinsics.columns.3.x),
            Double(extrinsics.columns.3.y),
            Double(extrinsics.columns.3.z)
        ]
        
        // Extract lens distortion

        var distortionCoefficients: [Float] = []
        if let lookupTable = calibrationData.lensDistortionLookupTable {
            lookupTable.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
                if let baseAddress = buffer.baseAddress {
                    let floatBuffer = baseAddress.assumingMemoryBound(to: Float.self)
                    let count = lookupTable.count / MemoryLayout<Float>.stride
                    distortionCoefficients = Array(UnsafeBufferPointer(start: floatBuffer, count: count))
                }
            }
            
            // Print LUT for debugging
            print("📋 Lens Distortion LUT (\(distortionCoefficients.count) entries):")
            print("   Range: [\(distortionCoefficients.min() ?? 0), \(distortionCoefficients.max() ?? 0)]")
            print("   First 5: \(distortionCoefficients.prefix(5))")
            print("   Last 5: \(distortionCoefficients.suffix(5))")
        }
        
        var inverseDistortionCoefficients: [Float] = []
        if let inverseLookupTable = calibrationData.inverseLensDistortionLookupTable {
            inverseLookupTable.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
                if let baseAddress = buffer.baseAddress {
                    let floatBuffer = baseAddress.assumingMemoryBound(to: Float.self)
                    let count = inverseLookupTable.count / MemoryLayout<Float>.stride
                    inverseDistortionCoefficients = Array(UnsafeBufferPointer(start: floatBuffer, count: count))
                }
            }
        }
        
        let base64Image = imageData.base64EncodedString()
        
        let response: [String: Any] = [
            "image_base64": base64Image,
            "camera_calibration": [
                "K": K,
                "R": R,  // Rotation matrix (camera pose)
                "t": t,  // Translation vector (camera position)
                "image_dimensions": [
                    "width": Int(dimensions.width),
                    "height": Int(dimensions.height)
                ],
                "lens_distortion_center": [
                    "x": Double(calibrationData.lensDistortionCenter.x),
                    "y": Double(calibrationData.lensDistortionCenter.y)
                ],
                "pixel_size_mm": Double(calibrationData.pixelSize),
                "lens_distortion_lookup_table": distortionCoefficients,
                "inverse_lens_distortion_lookup_table": inverseDistortionCoefficients,
            ],
            "metadata": [
                "timestamp": Date().timeIntervalSince1970,
                "orientation": photo.metadata[kCGImagePropertyOrientation as String] as? Int ?? 1,
                "calibration_source": "apple_native_depth_enabled"
            ]
        ]
        
        print("✅ NATIVE Calibration Data:")
        print("   Resolution: \(dimensions.width)×\(dimensions.height)")
        print("   Focal length: fx=\(K[0][0]), fy=\(K[1][1])")
        print("   Principal point: cx=\(K[0][2]), cy=\(K[1][2])")
        print("   Pixel size: \(calibrationData.pixelSize) mm")
        print("   Distortion LUT entries: \(distortionCoefficients.count)")
        print("   Intrinsics: \(intrinsics)")
        
        sendJSONResponse(response: response, completion: completion)
    }
    
    private func sendPhotoWithFallbackCalibration(imageData: Data, photo: AVCapturePhoto, completion: @escaping (HTTPResponse) -> Void) {
        guard let image = UIImage(data: imageData) else {
            let errorData = "Could not decode image".data(using: .utf8) ?? Data()
            completion(HTTPResponse(statusCode: 500, data: errorData, contentType: "text/plain"))
            return
        }
        
        let width = Int(image.size.width)
        let height = Int(image.size.height)
        
        let maxDim = Double(max(width, height))
        let fx = maxDim * 1.2
        let fy = maxDim * 1.2
        let cx = Double(width) / 2.0
        let cy = Double(height) / 2.0
        
        let K = [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]
        
        let base64Image = imageData.base64EncodedString()
        
        let response: [String: Any] = [
            "image_base64": base64Image,
            "camera_calibration": [
                "K": K,
                "image_dimensions": [
                    "width": width,
                    "height": height
                ],
                "lens_distortion_center": [
                    "x": cx,
                    "y": cy
                ],
                "pixel_size_mm": 0.0,
                "lens_distortion_lookup_table": []
            ],
            "metadata": [
                "timestamp": Date().timeIntervalSince1970,
                "orientation": photo.metadata[kCGImagePropertyOrientation as String] as? Int ?? 1,
                "calibration_source": "estimated_fallback",
                "warning": "No native calibration available. Using estimated intrinsics. For better accuracy, use a device with dual camera or perform manual calibration."
            ]
        ]
        
        print("⚠️ ESTIMATED Calibration (Fallback):")
        print("   Resolution: \(width)×\(height)")
        print("   Focal length (estimated): fx=\(fx), fy=\(fy)")
        print("   Principal point (estimated): cx=\(cx), cy=\(cy)")
        
        sendJSONResponse(response: response, completion: completion)
    }
    
    private func sendJSONResponse(response: [String: Any], completion: @escaping (HTTPResponse) -> Void) {
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: response, options: .prettyPrinted)
            print("   JSON size: \(jsonData.count) bytes")
            completion(HTTPResponse(statusCode: 200, data: jsonData, contentType: "application/json"))
            print("✅ Sent response to HTTP client")
        } catch {
            print("❌ Error encoding JSON: \(error)")
            let errorData = "Error encoding JSON: \(error.localizedDescription)".data(using: .utf8) ?? Data()
            completion(HTTPResponse(statusCode: 500, data: errorData, contentType: "text/plain"))
        }
    }
}
