import React, {useState, useEffect, useRef, useContext} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
  ActivityIndicator,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {SafeAreaView} from 'react-native-safe-area-context';
import Markdown from 'react-native-markdown-display';

import {ChatContext} from '../context/ChatContext';
import {AuthContext} from '../context/AuthContext';
import {ThemeContext} from '../context/ThemeContext';
import {ChatService} from '../services/ChatService';
import {Message, MessageType} from '../types/Chat';

interface ChatScreenProps {}

const ChatScreen: React.FC<ChatScreenProps> = () => {
  const {messages, addMessage, isLoading, setIsLoading} = useContext(ChatContext);
  const {user} = useContext(AuthContext);
  const {theme} = useContext(ThemeContext);
  
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    if (messages.length > 0) {
      flatListRef.current?.scrollToEnd({animated: true});
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: MessageType.USER,
      content: inputText.trim(),
      timestamp: new Date(),
      user: user?.name || 'User',
    };

    addMessage(userMessage);
    setInputText('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      const response = await ChatService.sendMessage({
        message: inputText.trim(),
        conversationId: 'default', // In a real app, manage conversation IDs
        stream: false,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: MessageType.ASSISTANT,
        content: response.message || 'Sorry, I could not process your request.',
        timestamp: new Date(),
        user: 'Assistant',
      };

      addMessage(assistantMessage);
    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: MessageType.SYSTEM,
        content: 'Failed to send message. Please check your connection and try again.',
        timestamp: new Date(),
        user: 'System',
      };

      addMessage(errorMessage);
      
      Alert.alert(
        'Error',
        'Failed to send message. Please check your connection and try again.',
      );
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  const renderMessage = ({item}: {item: Message}) => {
    const isUser = item.type === MessageType.USER;
    const isSystem = item.type === MessageType.SYSTEM;

    return (
      <View
        style={[
          styles.messageContainer,
          isUser ? styles.userMessage : styles.assistantMessage,
          isSystem && styles.systemMessage,
        ]}>
        <View
          style={[
            styles.messageBubble,
            isUser
              ? [styles.userBubble, {backgroundColor: theme.colors.primary}]
              : [styles.assistantBubble, {backgroundColor: theme.colors.surface}],
            isSystem && {backgroundColor: theme.colors.warning},
          ]}>
          {isUser ? (
            <Text
              style={[
                styles.messageText,
                {color: theme.colors.onPrimary},
              ]}>
              {item.content}
            </Text>
          ) : (
            <Markdown
              style={{
                body: {
                  color: theme.colors.onSurface,
                  fontSize: 16,
                },
                code_inline: {
                  backgroundColor: theme.colors.surfaceVariant,
                  color: theme.colors.onSurfaceVariant,
                },
                code_block: {
                  backgroundColor: theme.colors.surfaceVariant,
                  color: theme.colors.onSurfaceVariant,
                },
              }}>
              {item.content}
            </Markdown>
          )}
        </View>
        <Text style={[styles.timestamp, {color: theme.colors.outline}]}>
          {item.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </Text>
      </View>
    );
  };

  const renderTypingIndicator = () => {
    if (!isTyping) return null;

    return (
      <View style={[styles.messageContainer, styles.assistantMessage]}>
        <View
          style={[
            styles.messageBubble,
            styles.assistantBubble,
            {backgroundColor: theme.colors.surface},
          ]}>
          <View style={styles.typingIndicator}>
            <ActivityIndicator size="small" color={theme.colors.primary} />
            <Text style={[styles.typingText, {color: theme.colors.onSurface}]}>
              Assistant is typing...
            </Text>
          </View>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView
      style={[styles.container, {backgroundColor: theme.colors.background}]}>
      <View style={[styles.header, {backgroundColor: theme.colors.surface}]}>
        <Text style={[styles.headerTitle, {color: theme.colors.onSurface}]}>
          AI Assistant
        </Text>
        <TouchableOpacity style={styles.headerButton}>
          <Icon
            name="more-vert"
            size={24}
            color={theme.colors.onSurface}
          />
        </TouchableOpacity>
      </View>

      <KeyboardAvoidingView
        style={styles.chatContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}>
        
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={item => item.id}
          style={styles.messagesList}
          contentContainerStyle={styles.messagesContent}
          showsVerticalScrollIndicator={false}
          ListFooterComponent={renderTypingIndicator}
        />

        <View
          style={[
            styles.inputContainer,
            {backgroundColor: theme.colors.surface},
          ]}>
          <TextInput
            style={[
              styles.textInput,
              {
                backgroundColor: theme.colors.surfaceVariant,
                color: theme.colors.onSurfaceVariant,
              },
            ]}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type your message..."
            placeholderTextColor={theme.colors.outline}
            multiline
            maxLength={1000}
            editable={!isLoading}
          />
          
          <TouchableOpacity
            style={[
              styles.sendButton,
              {
                backgroundColor: inputText.trim() && !isLoading
                  ? theme.colors.primary
                  : theme.colors.outline,
              },
            ]}
            onPress={handleSendMessage}
            disabled={!inputText.trim() || isLoading}>
            {isLoading ? (
              <ActivityIndicator size="small" color={theme.colors.onPrimary} />
            ) : (
              <Icon
                name="send"
                size={20}
                color={theme.colors.onPrimary}
              />
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  headerButton: {
    padding: 8,
  },
  chatContainer: {
    flex: 1,
  },
  messagesList: {
    flex: 1,
  },
  messagesContent: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  messageContainer: {
    marginVertical: 4,
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  assistantMessage: {
    alignItems: 'flex-start',
  },
  systemMessage: {
    alignItems: 'center',
  },
  messageBubble: {
    maxWidth: '80%',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
  },
  userBubble: {
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  timestamp: {
    fontSize: 12,
    marginTop: 4,
    marginHorizontal: 16,
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  typingText: {
    marginLeft: 8,
    fontSize: 14,
    fontStyle: 'italic',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  textInput: {
    flex: 1,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginRight: 12,
    maxHeight: 100,
    fontSize: 16,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default ChatScreen;
