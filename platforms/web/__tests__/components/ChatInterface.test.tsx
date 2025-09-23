import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';
import ChatInterface from '../../components/ChatInterface';

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: WebSocket.OPEN,
};

// Mock global WebSocket
global.WebSocket = jest.fn(() => mockWebSocket) as any;

// Mock EventSource for SSE
const mockEventSource = {
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  close: jest.fn(),
};

global.EventSource = jest.fn(() => mockEventSource) as any;

// Mock fetch
global.fetch = jest.fn();

describe('ChatInterface', () => {
  const mockProps = {
    apiUrl: 'http://localhost:8080',
    tenantId: 'test-tenant',
    userId: 'test-user',
    onMessage: jest.fn(),
    onError: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders chat interface correctly', () => {
    render(<ChatInterface {...mockProps} />);
    
    expect(screen.getByPlaceholderText(/type your message/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /voice/i })).toBeInTheDocument();
  });

  it('displays welcome message on initial load', () => {
    render(<ChatInterface {...mockProps} />);
    
    expect(screen.getByText(/welcome/i)).toBeInTheDocument();
    expect(screen.getByText(/how can I help you today/i)).toBeInTheDocument();
  });

  it('handles text message input and submission', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const input = screen.getByPlaceholderText(/type your message/i);
    const sendButton = screen.getByRole('button', { name: /send/i });
    
    await user.type(input, 'Hello, how are you?');
    expect(input).toHaveValue('Hello, how are you?');
    
    await user.click(sendButton);
    
    // Should clear input after sending
    expect(input).toHaveValue('');
    
    // Should display user message
    expect(screen.getByText('Hello, how are you?')).toBeInTheDocument();
  });

  it('handles Enter key for message submission', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const input = screen.getByPlaceholderText(/type your message/i);
    
    await user.type(input, 'Test message');
    await user.keyboard('{Enter}');
    
    expect(input).toHaveValue('');
    expect(screen.getByText('Test message')).toBeInTheDocument();
  });

  it('prevents submission of empty messages', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const sendButton = screen.getByRole('button', { name: /send/i });
    
    await user.click(sendButton);
    
    // Should not create any message bubbles
    expect(screen.queryByTestId('message-bubble')).not.toBeInTheDocument();
  });

  it('handles Shift+Enter for new line', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const input = screen.getByPlaceholderText(/type your message/i);
    
    await user.type(input, 'Line 1');
    await user.keyboard('{Shift>}{Enter}{/Shift}');
    await user.type(input, 'Line 2');
    
    expect(input).toHaveValue('Line 1\nLine 2');
  });

  it('toggles voice mode correctly', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const voiceButton = screen.getByRole('button', { name: /voice/i });
    
    await user.click(voiceButton);
    
    // Should show voice recording UI
    expect(screen.getByText(/listening/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
  });

  it('displays typing indicator when bot is responding', async () => {
    render(<ChatInterface {...mockProps} />);
    
    // Simulate receiving a message that triggers typing indicator
    const messageEvent = new MessageEvent('message', {
      data: JSON.stringify({
        type: 'typing',
        isTyping: true
      })
    });
    
    // Trigger the event listener
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const typingHandler = addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];
    
    if (typingHandler) {
      typingHandler(messageEvent);
    }
    
    await waitFor(() => {
      expect(screen.getByText(/typing/i)).toBeInTheDocument();
    });
  });

  it('handles streaming message responses', async () => {
    render(<ChatInterface {...mockProps} />);
    
    // Simulate streaming response
    const streamingEvents = [
      { type: 'delta', content: 'Hello', seq: 1 },
      { type: 'delta', content: ' there!', seq: 2 },
      { type: 'done', seq: 3 }
    ];
    
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const messageHandler = addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];
    
    if (messageHandler) {
      for (const event of streamingEvents) {
        const messageEvent = new MessageEvent('message', {
          data: JSON.stringify(event)
        });
        messageHandler(messageEvent);
      }
    }
    
    await waitFor(() => {
      expect(screen.getByText('Hello there!')).toBeInTheDocument();
    });
  });

  it('handles message with references', async () => {
    render(<ChatInterface {...mockProps} />);
    
    const messageWithRefs = {
      type: 'message',
      content: 'Based on the documents, here is the answer...',
      references: [
        { title: 'Document 1', url: 'http://example.com/doc1' },
        { title: 'Document 2', url: 'http://example.com/doc2' }
      ]
    };
    
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const messageHandler = addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];
    
    if (messageHandler) {
      const messageEvent = new MessageEvent('message', {
        data: JSON.stringify(messageWithRefs)
      });
      messageHandler(messageEvent);
    }
    
    await waitFor(() => {
      expect(screen.getByText('Based on the documents, here is the answer...')).toBeInTheDocument();
      expect(screen.getByText('Document 1')).toBeInTheDocument();
      expect(screen.getByText('Document 2')).toBeInTheDocument();
    });
  });

  it('handles error messages gracefully', async () => {
    render(<ChatInterface {...mockProps} />);
    
    const errorEvent = new MessageEvent('error', {
      data: JSON.stringify({
        error: 'Connection failed',
        code: 'CONNECTION_ERROR'
      })
    });
    
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const errorHandler = addEventListener.mock.calls.find(
      call => call[0] === 'error'
    )?.[1];
    
    if (errorHandler) {
      errorHandler(errorEvent);
    }
    
    await waitFor(() => {
      expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
      expect(mockProps.onError).toHaveBeenCalledWith('Connection failed');
    });
  });

  it('handles voice recording and playback', async () => {
    // Mock MediaRecorder
    const mockMediaRecorder = {
      start: jest.fn(),
      stop: jest.fn(),
      addEventListener: jest.fn(),
      state: 'inactive'
    };
    
    global.MediaRecorder = jest.fn(() => mockMediaRecorder) as any;
    global.navigator.mediaDevices = {
      getUserMedia: jest.fn().mockResolvedValue(new MediaStream())
    } as any;
    
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const voiceButton = screen.getByRole('button', { name: /voice/i });
    await user.click(voiceButton);
    
    expect(mockMediaRecorder.start).toHaveBeenCalled();
    
    const stopButton = screen.getByRole('button', { name: /stop/i });
    await user.click(stopButton);
    
    expect(mockMediaRecorder.stop).toHaveBeenCalled();
  });

  it('displays conversation history correctly', () => {
    const messagesHistory = [
      { id: '1', role: 'user', content: 'Hello', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Hi there!', timestamp: new Date() },
      { id: '3', role: 'user', content: 'How are you?', timestamp: new Date() },
    ];
    
    render(<ChatInterface {...mockProps} initialMessages={messagesHistory} />);
    
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
    expect(screen.getByText('How are you?')).toBeInTheDocument();
  });

  it('auto-scrolls to bottom when new messages arrive', async () => {
    const mockScrollIntoView = jest.fn();
    Element.prototype.scrollIntoView = mockScrollIntoView;
    
    render(<ChatInterface {...mockProps} />);
    
    const input = screen.getByPlaceholderText(/type your message/i);
    const sendButton = screen.getByRole('button', { name: /send/i });
    
    await userEvent.type(input, 'Test message');
    await userEvent.click(sendButton);
    
    await waitFor(() => {
      expect(mockScrollIntoView).toHaveBeenCalled();
    });
  });

  it('handles connection reconnection', async () => {
    render(<ChatInterface {...mockProps} />);
    
    // Simulate connection error
    const errorEvent = new Event('error');
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const errorHandler = addEventListener.mock.calls.find(
      call => call[0] === 'error'
    )?.[1];
    
    if (errorHandler) {
      errorHandler(errorEvent);
    }
    
    await waitFor(() => {
      expect(screen.getByText(/reconnecting/i)).toBeInTheDocument();
    });
  });

  it('handles message retry functionality', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    // Send a message that will fail
    const input = screen.getByPlaceholderText(/type your message/i);
    await user.type(input, 'Test message');
    await user.click(screen.getByRole('button', { name: /send/i }));
    
    // Simulate error response
    const errorEvent = new MessageEvent('error', {
      data: JSON.stringify({ error: 'Failed to send message' })
    });
    
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const errorHandler = addEventListener.mock.calls.find(
      call => call[0] === 'error'
    )?.[1];
    
    if (errorHandler) {
      errorHandler(errorEvent);
    }
    
    await waitFor(() => {
      const retryButton = screen.getByRole('button', { name: /retry/i });
      expect(retryButton).toBeInTheDocument();
    });
    
    // Click retry
    const retryButton = screen.getByRole('button', { name: /retry/i });
    await user.click(retryButton);
    
    // Should attempt to resend the message
    expect(mockEventSource.addEventListener).toHaveBeenCalled();
  });

  it('handles markdown rendering in messages', async () => {
    render(<ChatInterface {...mockProps} />);
    
    const markdownMessage = {
      type: 'message',
      content: '**Bold text** and *italic text* with `code`'
    };
    
    const addEventListener = mockEventSource.addEventListener as jest.Mock;
    const messageHandler = addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )?.[1];
    
    if (messageHandler) {
      const messageEvent = new MessageEvent('message', {
        data: JSON.stringify(markdownMessage)
      });
      messageHandler(messageEvent);
    }
    
    await waitFor(() => {
      expect(screen.getByText('Bold text')).toBeInTheDocument();
      expect(screen.getByText('italic text')).toBeInTheDocument();
      expect(screen.getByText('code')).toBeInTheDocument();
    });
  });

  it('handles file upload functionality', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const fileInput = screen.getByLabelText(/upload file/i);
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    
    await user.upload(fileInput, file);
    
    expect(screen.getByText('test.txt')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /remove file/i })).toBeInTheDocument();
  });

  it('handles conversation clearing', async () => {
    const user = userEvent.setup();
    const messagesHistory = [
      { id: '1', role: 'user', content: 'Hello', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Hi there!', timestamp: new Date() },
    ];
    
    render(<ChatInterface {...mockProps} initialMessages={messagesHistory} />);
    
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
    
    const clearButton = screen.getByRole('button', { name: /clear conversation/i });
    await user.click(clearButton);
    
    // Should show confirmation dialog
    expect(screen.getByText(/are you sure/i)).toBeInTheDocument();
    
    const confirmButton = screen.getByRole('button', { name: /confirm/i });
    await user.click(confirmButton);
    
    // Messages should be cleared
    expect(screen.queryByText('Hello')).not.toBeInTheDocument();
    expect(screen.queryByText('Hi there!')).not.toBeInTheDocument();
  });

  it('handles keyboard shortcuts', async () => {
    const user = userEvent.setup();
    render(<ChatInterface {...mockProps} />);
    
    const input = screen.getByPlaceholderText(/type your message/i);
    
    // Test Ctrl+K for clearing conversation
    await user.keyboard('{Control>}k{/Control}');
    expect(screen.getByText(/clear conversation/i)).toBeInTheDocument();
    
    // Test Escape for canceling voice recording
    const voiceButton = screen.getByRole('button', { name: /voice/i });
    await user.click(voiceButton);
    
    await user.keyboard('{Escape}');
    expect(screen.queryByText(/listening/i)).not.toBeInTheDocument();
  });
});
