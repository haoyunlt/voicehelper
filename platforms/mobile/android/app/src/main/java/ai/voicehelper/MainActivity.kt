package ai.voicehelper

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import kotlin.random.Random

// 数据模型
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val content: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis()
)

data class Service(
    val id: String,
    val name: String,
    val description: String,
    val category: String,
    val isConnected: Boolean,
    val iconRes: Int? = null
)

data class ServiceCategory(
    val name: String,
    val services: List<Service>
)

// ViewModel
class ChatViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    fun addMessage(message: ChatMessage) {
        _messages.value = _messages.value + message
    }
    
    fun clearMessages() {
        _messages.value = emptyList()
    }
    
    suspend fun sendMessage(text: String) {
        val userMessage = ChatMessage(content = text, isUser = true)
        addMessage(userMessage)
        
        _isLoading.value = true
        
        // 模拟API调用
        delay(1000)
        
        val response = "这是对 '$text' 的智能回复。VoiceHelper AI 正在为您提供最佳的对话体验。"
        val assistantMessage = ChatMessage(content = response, isUser = false)
        addMessage(assistantMessage)
        
        _isLoading.value = false
    }
}

class VoiceViewModel : ViewModel() {
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    private val _transcribedText = MutableStateFlow("")
    val transcribedText: StateFlow<String> = _transcribedText.asStateFlow()
    
    fun startRecording() {
        _isRecording.value = true
        _transcribedText.value = ""
    }
    
    fun stopRecording() {
        _isRecording.value = false
        _isProcessing.value = true
        
        // 模拟语音识别
        viewModelScope.launch {
            delay(2000)
            _transcribedText.value = "这是语音识别的结果文本"
            _isProcessing.value = false
        }
    }
    
    fun clearTranscription() {
        _transcribedText.value = ""
    }
}

class ServicesViewModel : ViewModel() {
    private val _serviceCategories = MutableStateFlow<List<ServiceCategory>>(emptyList())
    val serviceCategories: StateFlow<List<ServiceCategory>> = _serviceCategories.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    init {
        loadServices()
    }
    
    private fun loadServices() {
        _isLoading.value = true
        
        // 模拟服务数据
        val services = listOf(
            Service("1", "微信", "微信消息发送和接收", "社交平台", true),
            Service("2", "钉钉", "钉钉工作通知和审批", "办公套件", false),
            Service("3", "GitHub", "代码仓库管理和CI/CD", "开发工具", true),
            Service("4", "淘宝", "商品搜索和订单管理", "电商平台", false),
            Service("5", "阿里云", "云资源管理和监控", "云服务", true),
            Service("6", "OpenAI", "GPT模型调用和管理", "AI/ML", true)
        )
        
        val categories = services.groupBy { it.category }.map { (category, services) ->
            ServiceCategory(category, services)
        }
        
        _serviceCategories.value = categories
        _isLoading.value = false
    }
    
    fun refreshServices() {
        loadServices()
    }
}

// 主Activity
class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    
    private lateinit var textToSpeech: TextToSpeech
    private lateinit var speechRecognizer: SpeechRecognizer
    private var isPermissionGranted = false
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        isPermissionGranted = permissions[Manifest.permission.RECORD_AUDIO] == true
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 初始化TTS
        textToSpeech = TextToSpeech(this, this)
        
        // 初始化语音识别
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        
        // 请求权限
        requestPermissions()
        
        setContent {
            VoiceHelperTheme {
                VoiceHelperApp(
                    onSpeak = { text -> speak(text) },
                    onStartRecording = { startVoiceRecognition() },
                    onStopRecording = { stopVoiceRecognition() }
                )
            }
        }
    }
    
    private fun requestPermissions() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                isPermissionGranted = true
            }
            else -> {
                requestPermissionLauncher.launch(
                    arrayOf(
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.INTERNET
                    )
                )
            }
        }
    }
    
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech.language = Locale.CHINESE
        }
    }
    
    private fun speak(text: String) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }
    
    private fun startVoiceRecognition() {
        if (!isPermissionGranted) {
            requestPermissions()
            return
        }
        
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
        
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {}
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.firstOrNull()?.let { result ->
                    // 处理识别结果
                }
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        
        speechRecognizer.startListening(intent)
    }
    
    private fun stopVoiceRecognition() {
        speechRecognizer.stopListening()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        textToSpeech.shutdown()
        speechRecognizer.destroy()
    }
}

// 主应用组合
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VoiceHelperApp(
    onSpeak: (String) -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit
) {
    var selectedTab by remember { mutableStateOf(0) }
    
    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    icon = { Icon(Icons.Default.Chat, contentDescription = "对话") },
                    label = { Text("对话") },
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Default.Mic, contentDescription = "语音") },
                    label = { Text("语音") },
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Default.Apps, contentDescription = "服务") },
                    label = { Text("服务") },
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Default.Settings, contentDescription = "设置") },
                    label = { Text("设置") },
                    selected = selectedTab == 3,
                    onClick = { selectedTab = 3 }
                )
            }
        }
    ) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            when (selectedTab) {
                0 -> ChatScreen(onSpeak = onSpeak)
                1 -> VoiceScreen(
                    onStartRecording = onStartRecording,
                    onStopRecording = onStopRecording
                )
                2 -> ServicesScreen()
                3 -> SettingsScreen()
            }
        }
    }
}

// 对话界面
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    chatViewModel: ChatViewModel = viewModel(),
    onSpeak: (String) -> Unit
) {
    val messages by chatViewModel.messages.collectAsState()
    val isLoading by chatViewModel.isLoading.collectAsState()
    var messageText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()
    
    // 自动滚动到最新消息
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }
    
    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // 顶部栏
        TopAppBar(
            title = { Text("VoiceHelper") },
            actions = {
                IconButton(onClick = { chatViewModel.clearMessages() }) {
                    Icon(Icons.Default.Clear, contentDescription = "清空")
                }
            }
        )
        
        // 消息列表
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(messages) { message ->
                MessageBubble(
                    message = message,
                    onSpeak = onSpeak
                )
            }
            
            if (isLoading) {
                item {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Start
                    ) {
                        TypingIndicator()
                    }
                }
            }
        }
        
        // 输入区域
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = messageText,
                    onValueChange = { messageText = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("输入消息...") },
                    maxLines = 3
                )
                
                Spacer(modifier = Modifier.width(8.dp))
                
                IconButton(
                    onClick = {
                        if (messageText.isNotBlank()) {
                            coroutineScope.launch {
                                chatViewModel.sendMessage(messageText)
                            }
                            messageText = ""
                        }
                    },
                    enabled = messageText.isNotBlank() && !isLoading
                ) {
                    Icon(Icons.Default.Send, contentDescription = "发送")
                }
            }
        }
    }
}

// 消息气泡
@Composable
fun MessageBubble(
    message: ChatMessage,
    onSpeak: (String) -> Unit
) {
    val timeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())
    
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!message.isUser) {
            // AI头像
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.SmartToy,
                    contentDescription = "AI",
                    tint = Color.White,
                    modifier = Modifier.size(24.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(8.dp))
        }
        
        Column(
            modifier = Modifier.widthIn(max = 280.dp),
            horizontalAlignment = if (message.isUser) Alignment.End else Alignment.Start
        ) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = if (message.isUser) 
                        MaterialTheme.colorScheme.primary 
                    else 
                        MaterialTheme.colorScheme.surfaceVariant
                ),
                shape = RoundedCornerShape(
                    topStart = 16.dp,
                    topEnd = 16.dp,
                    bottomStart = if (message.isUser) 16.dp else 4.dp,
                    bottomEnd = if (message.isUser) 4.dp else 16.dp
                )
            ) {
                Text(
                    text = message.content,
                    modifier = Modifier.padding(12.dp),
                    color = if (message.isUser) Color.White else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
            ) {
                Text(
                    text = timeFormat.format(Date(message.timestamp)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                if (!message.isUser) {
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = { onSpeak(message.content) },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            Icons.Default.VolumeUp,
                            contentDescription = "播放",
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
        }
        
        if (message.isUser) {
            Spacer(modifier = Modifier.width(8.dp))
            
            // 用户头像
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.secondary),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Person,
                    contentDescription = "用户",
                    tint = Color.White,
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    }
}

// 输入指示器
@Composable
fun TypingIndicator() {
    val infiniteTransition = rememberInfiniteTransition()
    
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        // AI头像
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Default.SmartToy,
                contentDescription = "AI",
                tint = Color.White,
                modifier = Modifier.size(24.dp)
            )
        }
        
        Spacer(modifier = Modifier.width(8.dp))
        
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
            shape = RoundedCornerShape(16.dp)
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                repeat(3) { index ->
                    val animatedAlpha by infiniteTransition.animateFloat(
                        initialValue = 0.3f,
                        targetValue = 1f,
                        animationSpec = infiniteRepeatable(
                            animation = tween(600),
                            repeatMode = RepeatMode.Reverse
                        )
                    )
                    
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(
                                MaterialTheme.colorScheme.onSurfaceVariant.copy(
                                    alpha = if (index == 0) animatedAlpha else 
                                           if (index == 1) animatedAlpha * 0.7f else 
                                           animatedAlpha * 0.4f
                                )
                            )
                    )
                    
                    if (index < 2) {
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                }
            }
        }
    }
}

// 语音界面
@Composable
fun VoiceScreen(
    voiceViewModel: VoiceViewModel = viewModel(),
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit
) {
    val isRecording by voiceViewModel.isRecording.collectAsState()
    val isProcessing by voiceViewModel.isProcessing.collectAsState()
    val transcribedText by voiceViewModel.transcribedText.collectAsState()
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // 语音波形动画
        VoiceWaveAnimation(isAnimating = isRecording)
        
        Spacer(modifier = Modifier.height(32.dp))
        
        // 状态文本
        Text(
            text = when {
                isRecording -> "正在聆听..."
                isProcessing -> "处理中..."
                else -> "点击开始语音对话"
            },
            style = MaterialTheme.typography.headlineSmall,
            textAlign = TextAlign.Center
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // 转录文本
        if (transcribedText.isNotEmpty()) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = transcribedText,
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }
        
        Spacer(modifier = Modifier.height(32.dp))
        
        // 录音按钮
        val scale by animateFloatAsState(
            targetValue = if (isRecording) 1.1f else 1f,
            animationSpec = tween(150)
        )
        
        FloatingActionButton(
            onClick = {
                if (isRecording) {
                    voiceViewModel.stopRecording()
                    onStopRecording()
                } else {
                    voiceViewModel.startRecording()
                    onStartRecording()
                }
            },
            modifier = Modifier
                .size(80.dp)
                .scale(scale),
            containerColor = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
        ) {
            Icon(
                if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                contentDescription = if (isRecording) "停止" else "开始",
                modifier = Modifier.size(32.dp),
                tint = Color.White
            )
        }
    }
}

// 语音波形动画
@Composable
fun VoiceWaveAnimation(isAnimating: Boolean) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(20) { index ->
            val infiniteTransition = rememberInfiniteTransition()
            
            val height by infiniteTransition.animateFloat(
                initialValue = 20f,
                targetValue = if (isAnimating) Random.nextFloat() * 60 + 20 else 20f,
                animationSpec = infiniteRepeatable(
                    animation = tween(
                        durationMillis = 300 + index * 50,
                        easing = LinearEasing
                    ),
                    repeatMode = RepeatMode.Reverse
                )
            )
            
            Box(
                modifier = Modifier
                    .width(4.dp)
                    .height(height.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(
                        if (isAnimating) 
                            MaterialTheme.colorScheme.primary 
                        else 
                            MaterialTheme.colorScheme.outline
                    )
            )
        }
    }
}

// 服务界面
@Composable
fun ServicesScreen(
    servicesViewModel: ServicesViewModel = viewModel()
) {
    val serviceCategories by servicesViewModel.serviceCategories.collectAsState()
    val isLoading by servicesViewModel.isLoading.collectAsState()
    
    Column(modifier = Modifier.fillMaxSize()) {
        // 顶部栏
        TopAppBar(
            title = { Text("服务集成") },
            actions = {
                IconButton(onClick = { servicesViewModel.refreshServices() }) {
                    Icon(Icons.Default.Refresh, contentDescription = "刷新")
                }
            }
        )
        
        if (isLoading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        } else {
            LazyColumn(
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                serviceCategories.forEach { category ->
                    item {
                        Text(
                            text = category.name,
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    }
                    
                    items(category.services) { service ->
                        ServiceCard(service = service)
                    }
                }
            }
        }
    }
}

// 服务卡片
@Composable
fun ServiceCard(service: Service) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 服务图标
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Apps,
                    contentDescription = service.name,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            // 服务信息
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = service.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium
                )
                
                Text(
                    text = service.description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            // 连接状态
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .clip(CircleShape)
                    .background(
                        if (service.isConnected) 
                            Color.Green 
                        else 
                            MaterialTheme.colorScheme.outline
                    )
            )
        }
    }
}

// 设置界面
@Composable
fun SettingsScreen() {
    var apiKey by remember { mutableStateOf("") }
    var voiceEnabled by remember { mutableStateOf(true) }
    var autoSpeak by remember { mutableStateOf(false) }
    var showApiKeyDialog by remember { mutableStateOf(false) }
    
    Column(modifier = Modifier.fillMaxSize()) {
        // 顶部栏
        TopAppBar(
            title = { Text("设置") }
        )
        
        LazyColumn(
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // API配置
            item {
                Card {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "API 配置",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("API Key")
                            
                            TextButton(onClick = { showApiKeyDialog = true }) {
                                Text(if (apiKey.isEmpty()) "设置" else "已设置")
                            }
                        }
                    }
                }
            }
            
            // 语音设置
            item {
                Card {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "语音设置",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("启用语音功能")
                            Switch(
                                checked = voiceEnabled,
                                onCheckedChange = { voiceEnabled = it }
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("自动播放回复")
                            Switch(
                                checked = autoSpeak,
                                onCheckedChange = { autoSpeak = it }
                            )
                        }
                    }
                }
            }
            
            // 关于
            item {
                Card {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "关于",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("版本")
                            Text(
                                "1.9.0",
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        TextButton(
                            onClick = { /* 打开文档 */ },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("开发者文档")
                        }
                        
                        TextButton(
                            onClick = { /* 打开隐私政策 */ },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("隐私政策")
                        }
                    }
                }
            }
        }
    }
    
    // API Key 对话框
    if (showApiKeyDialog) {
        AlertDialog(
            onDismissRequest = { showApiKeyDialog = false },
            title = { Text("API Key") },
            text = {
                Column {
                    Text("请输入您的 VoiceHelper API Key")
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = apiKey,
                        onValueChange = { apiKey = it },
                        placeholder = { Text("输入 API Key") },
                        singleLine = true
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showApiKeyDialog = false }) {
                    Text("确定")
                }
            },
            dismissButton = {
                TextButton(onClick = { showApiKeyDialog = false }) {
                    Text("取消")
                }
            }
        )
    }
}

// 主题
@Composable
fun VoiceHelperTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Color(0xFF2196F3),
            secondary = Color(0xFF03DAC6),
            background = Color(0xFFF5F5F5)
        ),
        content = content
    )
}
