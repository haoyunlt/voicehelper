import SwiftUI
import AVFoundation
import Speech
import Combine

// MARK: - 主视图
struct ContentView: View {
    @StateObject private var voiceManager = VoiceManager()
    @StateObject private var chatManager = ChatManager()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // 对话页面
            ChatView(chatManager: chatManager, voiceManager: voiceManager)
                .tabItem {
                    Image(systemName: "message.circle")
                    Text("对话")
                }
                .tag(0)
            
            // 语音页面
            VoiceView(voiceManager: voiceManager)
                .tabItem {
                    Image(systemName: "mic.circle")
                    Text("语音")
                }
                .tag(1)
            
            // 服务页面
            ServicesView()
                .tabItem {
                    Image(systemName: "app.connected.to.app.below.fill")
                    Text("服务")
                }
                .tag(2)
            
            // 设置页面
            SettingsView()
                .tabItem {
                    Image(systemName: "gear.circle")
                    Text("设置")
                }
                .tag(3)
        }
        .accentColor(.blue)
    }
}

// MARK: - 对话视图
struct ChatView: View {
    @ObservedObject var chatManager: ChatManager
    @ObservedObject var voiceManager: VoiceManager
    @State private var messageText = ""
    @State private var isRecording = false
    
    var body: some View {
        NavigationView {
            VStack {
                // 消息列表
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(chatManager.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: chatManager.messages.count) { _ in
                        if let lastMessage = chatManager.messages.last {
                            withAnimation {
                                proxy.scrollTo(lastMessage.id, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // 输入区域
                HStack {
                    // 文本输入
                    TextField("输入消息...", text: $messageText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .onSubmit {
                            sendMessage()
                        }
                    
                    // 语音按钮
                    Button(action: toggleRecording) {
                        Image(systemName: isRecording ? "mic.fill" : "mic")
                            .foregroundColor(isRecording ? .red : .blue)
                            .font(.title2)
                    }
                    .disabled(voiceManager.isProcessing)
                    
                    // 发送按钮
                    Button(action: sendMessage) {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(.blue)
                            .font(.title2)
                    }
                    .disabled(messageText.isEmpty)
                }
                .padding()
            }
            .navigationTitle("VoiceHelper")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("清空") {
                        chatManager.clearMessages()
                    }
                }
            }
        }
        .onReceive(voiceManager.$transcribedText) { text in
            if !text.isEmpty {
                messageText = text
                voiceManager.transcribedText = ""
            }
        }
    }
    
    private func sendMessage() {
        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = ChatMessage(
            id: UUID(),
            content: messageText,
            isUser: true,
            timestamp: Date()
        )
        
        chatManager.addMessage(userMessage)
        
        Task {
            await chatManager.sendMessage(messageText)
        }
        
        messageText = ""
    }
    
    private func toggleRecording() {
        if isRecording {
            voiceManager.stopRecording()
        } else {
            voiceManager.startRecording()
        }
        isRecording.toggle()
    }
}

// MARK: - 消息气泡
struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
                VStack(alignment: .trailing) {
                    Text(message.content)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    
                    Text(formatTime(message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            } else {
                VStack(alignment: .leading) {
                    HStack {
                        Image(systemName: "brain.head.profile")
                            .foregroundColor(.blue)
                        Text("VoiceHelper")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Text(message.content)
                        .padding()
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    
                    HStack {
                        Text(formatTime(message.timestamp))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        
                        if !message.content.isEmpty {
                            Button(action: {
                                // 播放语音
                                VoiceManager.shared.speak(text: message.content)
                            }) {
                                Image(systemName: "speaker.2")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                            }
                        }
                    }
                }
                Spacer()
            }
        }
    }
    
    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - 语音视图
struct VoiceView: View {
    @ObservedObject var voiceManager: VoiceManager
    @State private var isListening = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                Spacer()
                
                // 语音波形动画
                VoiceWaveView(isAnimating: voiceManager.isRecording)
                    .frame(height: 100)
                
                // 状态文本
                Text(voiceManager.isRecording ? "正在聆听..." : 
                     voiceManager.isProcessing ? "处理中..." : "点击开始语音对话")
                    .font(.title2)
                    .foregroundColor(.secondary)
                
                // 转录文本
                if !voiceManager.transcribedText.isEmpty {
                    Text(voiceManager.transcribedText)
                        .padding()
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal)
                }
                
                Spacer()
                
                // 录音按钮
                Button(action: toggleRecording) {
                    ZStack {
                        Circle()
                            .fill(voiceManager.isRecording ? Color.red : Color.blue)
                            .frame(width: 80, height: 80)
                        
                        Image(systemName: voiceManager.isRecording ? "stop.fill" : "mic.fill")
                            .font(.largeTitle)
                            .foregroundColor(.white)
                    }
                }
                .disabled(voiceManager.isProcessing)
                .scaleEffect(voiceManager.isRecording ? 1.1 : 1.0)
                .animation(.easeInOut(duration: 0.1), value: voiceManager.isRecording)
                
                Spacer()
            }
            .navigationTitle("语音助手")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    private func toggleRecording() {
        if voiceManager.isRecording {
            voiceManager.stopRecording()
        } else {
            voiceManager.startRecording()
        }
    }
}

// MARK: - 语音波形视图
struct VoiceWaveView: View {
    let isAnimating: Bool
    @State private var animationOffset: CGFloat = 0
    
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<20) { index in
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.blue.opacity(0.7))
                    .frame(width: 4, height: waveHeight(for: index))
                    .animation(
                        isAnimating ? 
                        Animation.easeInOut(duration: 0.5)
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.1) : 
                        .default,
                        value: isAnimating
                    )
            }
        }
    }
    
    private func waveHeight(for index: Int) -> CGFloat {
        if isAnimating {
            return CGFloat.random(in: 20...80)
        } else {
            return 20
        }
    }
}

// MARK: - 服务视图
struct ServicesView: View {
    @StateObject private var servicesManager = ServicesManager()
    
    var body: some View {
        NavigationView {
            List {
                ForEach(servicesManager.serviceCategories, id: \.name) { category in
                    Section(category.name) {
                        ForEach(category.services, id: \.id) { service in
                            ServiceRow(service: service)
                        }
                    }
                }
            }
            .navigationTitle("服务集成")
            .refreshable {
                await servicesManager.loadServices()
            }
        }
        .onAppear {
            Task {
                await servicesManager.loadServices()
            }
        }
    }
}

// MARK: - 服务行
struct ServiceRow: View {
    let service: Service
    
    var body: some View {
        HStack {
            // 服务图标
            AsyncImage(url: URL(string: service.iconURL)) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } placeholder: {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.3))
            }
            .frame(width: 40, height: 40)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(service.name)
                    .font(.headline)
                
                Text(service.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            // 状态指示器
            Circle()
                .fill(service.isConnected ? Color.green : Color.gray)
                .frame(width: 12, height: 12)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - 设置视图
struct SettingsView: View {
    @AppStorage("apiKey") private var apiKey = ""
    @AppStorage("voiceEnabled") private var voiceEnabled = true
    @AppStorage("autoSpeak") private var autoSpeak = false
    @State private var showingAPIKeyAlert = false
    
    var body: some View {
        NavigationView {
            Form {
                Section("API 配置") {
                    HStack {
                        Text("API Key")
                        Spacer()
                        Button(apiKey.isEmpty ? "设置" : "已设置") {
                            showingAPIKeyAlert = true
                        }
                        .foregroundColor(.blue)
                    }
                }
                
                Section("语音设置") {
                    Toggle("启用语音功能", isOn: $voiceEnabled)
                    Toggle("自动播放回复", isOn: $autoSpeak)
                }
                
                Section("关于") {
                    HStack {
                        Text("版本")
                        Spacer()
                        Text("1.9.0")
                            .foregroundColor(.secondary)
                    }
                    
                    Link("开发者文档", destination: URL(string: "https://docs.voicehelper.ai")!)
                    Link("隐私政策", destination: URL(string: "https://voicehelper.ai/privacy")!)
                }
            }
            .navigationTitle("设置")
        }
        .alert("API Key", isPresented: $showingAPIKeyAlert) {
            TextField("输入 API Key", text: $apiKey)
            Button("确定") { }
            Button("取消", role: .cancel) { }
        } message: {
            Text("请输入您的 VoiceHelper API Key")
        }
    }
}

// MARK: - 数据模型
struct ChatMessage: Identifiable, Codable {
    let id: UUID
    let content: String
    let isUser: Bool
    let timestamp: Date
}

struct Service: Identifiable, Codable {
    let id: String
    let name: String
    let description: String
    let iconURL: String
    let category: String
    let isConnected: Bool
}

struct ServiceCategory {
    let name: String
    let services: [Service]
}

// MARK: - 管理器类
class ChatManager: ObservableObject {
    @Published var messages: [ChatMessage] = []
    private let apiClient = VoiceHelperAPIClient()
    
    func addMessage(_ message: ChatMessage) {
        DispatchQueue.main.async {
            self.messages.append(message)
        }
    }
    
    func clearMessages() {
        DispatchQueue.main.async {
            self.messages.removeAll()
        }
    }
    
    func sendMessage(_ text: String) async {
        do {
            let response = await apiClient.sendMessage(text)
            let assistantMessage = ChatMessage(
                id: UUID(),
                content: response,
                isUser: false,
                timestamp: Date()
            )
            
            DispatchQueue.main.async {
                self.addMessage(assistantMessage)
            }
        } catch {
            let errorMessage = ChatMessage(
                id: UUID(),
                content: "抱歉，发生了错误：\(error.localizedDescription)",
                isUser: false,
                timestamp: Date()
            )
            
            DispatchQueue.main.async {
                self.addMessage(errorMessage)
            }
        }
    }
}

class VoiceManager: ObservableObject {
    static let shared = VoiceManager()
    
    @Published var isRecording = false
    @Published var isProcessing = false
    @Published var transcribedText = ""
    
    private var audioEngine = AVAudioEngine()
    private var speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let synthesizer = AVSpeechSynthesizer()
    private let apiClient = VoiceHelperAPIClient()
    
    init() {
        requestPermissions()
    }
    
    private func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { status in
            // 处理授权状态
        }
        
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            // 处理录音权限
        }
    }
    
    func startRecording() {
        guard !isRecording else { return }
        
        isRecording = true
        isProcessing = false
        transcribedText = ""
        
        // 配置音频会话
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        // 开始语音识别
        startSpeechRecognition()
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        isRecording = false
        isProcessing = true
        
        audioEngine.stop()
        recognitionRequest?.endAudio()
        
        // 处理最终的转录结果
        if !transcribedText.isEmpty {
            Task {
                await processTranscribedText()
            }
        }
    }
    
    private func startSpeechRecognition() {
        recognitionTask?.cancel()
        recognitionTask = nil
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        
        recognitionRequest.shouldReportPartialResults = true
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            if let result = result {
                DispatchQueue.main.async {
                    self?.transcribedText = result.bestTranscription.formattedString
                }
            }
            
            if error != nil || result?.isFinal == true {
                self?.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                self?.recognitionRequest = nil
                self?.recognitionTask = nil
            }
        }
    }
    
    private func processTranscribedText() async {
        // 这里可以调用 API 进行进一步处理
        isProcessing = false
    }
    
    func speak(text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = 0.5
        
        synthesizer.speak(utterance)
    }
}

class ServicesManager: ObservableObject {
    @Published var serviceCategories: [ServiceCategory] = []
    private let apiClient = VoiceHelperAPIClient()
    
    func loadServices() async {
        do {
            let services = await apiClient.getServices()
            let categories = Dictionary(grouping: services, by: { $0.category })
            
            DispatchQueue.main.async {
                self.serviceCategories = categories.map { category, services in
                    ServiceCategory(name: category, services: services)
                }.sorted { $0.name < $1.name }
            }
        } catch {
            print("Failed to load services: \(error)")
        }
    }
}

// MARK: - API 客户端
class VoiceHelperAPIClient {
    private let baseURL = "https://api.voicehelper.ai/v1"
    private var apiKey: String {
        UserDefaults.standard.string(forKey: "apiKey") ?? ""
    }
    
    func sendMessage(_ text: String) async -> String {
        // 模拟 API 调用
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        return "这是对 '\(text)' 的回复"
    }
    
    func getServices() async -> [Service] {
        // 模拟服务列表
        return [
            Service(id: "1", name: "微信", description: "微信消息发送和接收", iconURL: "", category: "社交平台", isConnected: true),
            Service(id: "2", name: "钉钉", description: "钉钉工作通知和审批", iconURL: "", category: "办公套件", isConnected: false),
            Service(id: "3", name: "GitHub", description: "代码仓库管理", iconURL: "", category: "开发工具", isConnected: true)
        ]
    }
}

// MARK: - 预览
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
