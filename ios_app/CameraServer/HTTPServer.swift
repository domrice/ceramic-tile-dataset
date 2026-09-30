import Foundation
import Network

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data
}

struct HTTPResponse {
    let statusCode: Int
    let data: Data
    let contentType: String
    
    init(statusCode: Int, data: Data, contentType: String = "application/json") {
        self.statusCode = statusCode
        self.data = data
        self.contentType = contentType
    }
}

class HTTPServer {
    private var listener: NWListener?
    private var handlers: [String: (HTTPRequest, @escaping (HTTPResponse) -> Void) -> Void] = [:]
    
    func addHandler(path: String, method: String, handler: @escaping (HTTPRequest, @escaping (HTTPResponse) -> Void) -> Void) {
        let key = "\(method) \(path)"
        handlers[key] = handler
    }
    
    func start(port: UInt16) {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        
        do {
            listener = try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: port)!)
            listener?.newConnectionHandler = { [weak self] connection in
                self?.handleConnection(connection)
            }
            
            listener?.start(queue: .global())
            print("Server started on port \(port)")
        } catch {
            print("Failed to start server: \(error)")
        }
    }
    
    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global())
        
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            if let data = data, !data.isEmpty {
                self?.processRequest(data: data, connection: connection)
            }
            
            if error != nil || isComplete {
                connection.cancel()
            }
        }
    }
    
    private func processRequest(data: Data, connection: NWConnection) {
        guard let requestString = String(data: data, encoding: .utf8) else {
            print("❌   Could not decode request")
            connection.cancel()
            return
        }
        
        print("Received request:\n\(requestString.prefix(200))")
        
        let lines = requestString.components(separatedBy: "\r\n")
        guard let firstLine = lines.first else {
            print("❌   No first line in request")
            connection.cancel()
            return
        }
        
        let components = firstLine.components(separatedBy: " ")
        guard components.count >= 3 else {
            print("❌   Invalid request line: \(firstLine)")
            connection.cancel()
            return
        }
        
        let method = components[0]
        let path = components[1]
        
        print("Processing: \(method) \(path)")
        
        let request = HTTPRequest(method: method, path: path, headers: [:], body: Data())
        let key = "\(method) \(path)"
        
        if let handler = handlers[key] {
            handler(request) { response in
                print("Sending response: \(response.statusCode)")
                self.sendResponse(connection: connection, response: response)
            }
        } else {
            print("❌   No handler for: \(key)")
            let notFoundResponse = HTTPResponse(statusCode: 404, data: "Not Found".data(using: .utf8) ?? Data(), contentType: "text/plain")
            sendResponse(connection: connection, response: notFoundResponse)
        }
    }
    
    private func sendResponse(connection: NWConnection, response: HTTPResponse) {
        let statusText = getStatusText(for: response.statusCode)
        
        let responseHeader = "HTTP/1.1 \(response.statusCode) \(statusText)\r\nContent-Type: \(response.contentType)\r\nContent-Length: \(response.data.count)\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
        
        guard let headerData = responseHeader.data(using: .utf8) else {
            print("❌   Could not encode response header")
            connection.cancel()
            return
        }
        
        let fullResponse = headerData + response.data
        
        print("📤 Response size: \(fullResponse.count) bytes")
        print("📤 Response header:\n\(responseHeader)")
        
        connection.send(content: fullResponse, completion: .contentProcessed { error in
            if let error = error {
                print("❌   Send error: \(error)")
            } else {
                print("✅   Response sent successfully")
            }
            connection.cancel()
        })
    }
    
    private func getStatusText(for code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        default: return "Unknown"
        }
    }
}
