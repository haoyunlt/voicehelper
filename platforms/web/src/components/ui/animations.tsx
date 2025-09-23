// UI动画组件库 - v1.2.0 (简化版)
import React, { useEffect, useState } from 'react';

// 简化的 cn 函数
const cn = (...classes: (string | undefined)[]) => {
  return classes.filter(Boolean).join(' ');
};

// ==================== 打字机效果 ====================
interface TypewriterProps {
  text: string;
  speed?: number;
  className?: string;
  onComplete?: () => void;
}

export const Typewriter: React.FC<TypewriterProps> = ({
  text,
  speed = 30,
  className,
  onComplete
}) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, speed);
      return () => clearTimeout(timeout);
    } else if (onComplete) {
      onComplete();
    }
    return undefined;
  }, [currentIndex, text, speed, onComplete]);

  return (
    <span className={className}>
      {displayText}
      {currentIndex < text.length && (
        <span className="inline-block w-0.5 h-4 bg-current ml-0.5 animate-pulse" />
      )}
    </span>
  );
};

// ==================== 语音波形动画 ====================
interface VoiceWaveProps {
  isActive: boolean;
  amplitude?: number[];
  className?: string;
}

export const VoiceWave: React.FC<VoiceWaveProps> = ({
  isActive,
  className
}) => {
  return (
    <div className={cn("flex items-center gap-1 h-8", className)}>
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          className={cn(
            "w-1 bg-gradient-to-t from-blue-500 to-blue-300 rounded-full transition-all duration-200",
            isActive ? "h-6 opacity-100 animate-pulse" : "h-2 opacity-30"
          )}
        />
      ))}
    </div>
  );
};

// ==================== 思考动画 ====================
interface ThinkingDotsProps {
  isVisible: boolean;
  className?: string;
}

export const ThinkingDots: React.FC<ThinkingDotsProps> = ({
  isVisible,
  className
}) => {
  if (!isVisible) return null;

  return (
    <div className={cn("flex gap-1", className)}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  );
};

// ==================== 消息气泡动画 ====================
interface MessageBubbleProps {
  children: React.ReactNode;
  role: 'user' | 'assistant';
  isNew?: boolean;
  className?: string;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  children,
  role,
  isNew = false,
  className
}) => {
  const isUser = role === 'user';

  return (
    <div
      className={cn(
        "max-w-[70%] p-4 rounded-2xl shadow-sm transition-all duration-300",
        isUser ? "bg-blue-500 text-white ml-auto" : "bg-gray-100 text-gray-800",
        isNew ? "animate-fade-in" : "",
        className
      )}
    >
      {children}
    </div>
  );
};

// ==================== 脉冲动画 ====================
interface PulseProps {
  isActive: boolean;
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  className?: string;
}

export const Pulse: React.FC<PulseProps> = ({
  isActive,
  size = 'md',
  color = 'blue',
  className
}) => {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-4 h-4',
    lg: 'w-6 h-6'
  };

  if (!isActive) return null;

  return (
    <div className={cn("relative", className)}>
      <div
        className={cn(
          sizeClasses[size],
          `bg-${color}-500 rounded-full animate-ping`
        )}
      />
      <div
        className={cn(
          sizeClasses[size],
          `bg-${color}-500 rounded-full absolute top-0 left-0`
        )}
      />
    </div>
  );
};

// ==================== 骨架屏 ====================
interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  variant?: 'text' | 'rect' | 'circle';
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = 20,
  variant = 'rect',
  className
}) => {
  const variantClasses = {
    text: 'rounded',
    rect: 'rounded-md',
    circle: 'rounded-full'
  };

  return (
    <div
      className={cn(
        "bg-gray-200 animate-pulse",
        variantClasses[variant],
        className
      )}
      style={{ width, height }}
    />
  );
};

// ==================== 成功/错误提示 ====================
interface ToastProps {
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  isVisible: boolean;
  onClose?: () => void;
}

export const Toast: React.FC<ToastProps> = ({
  message,
  type,
  isVisible,
  onClose
}) => {
  const colors = {
    success: 'bg-green-500',
    error: 'bg-red-500',
    warning: 'bg-yellow-500',
    info: 'bg-blue-500'
  };

  const icons = {
    success: '✓',
    error: '✕',
    warning: '!',
    info: 'i'
  };

  useEffect(() => {
    if (isVisible && onClose) {
      const timer = setTimeout(onClose, 3000);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [isVisible, onClose]);

  if (!isVisible) return null;

  return (
    <div
      className={cn(
        "fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-3 rounded-lg text-white shadow-lg transition-all duration-300",
        colors[type]
      )}
    >
      <span className="text-xl">{icons[type]}</span>
      <span>{message}</span>
      {onClose && (
        <button
          onClick={onClose}
          className="ml-2 hover:opacity-80"
        >
          ✕
        </button>
      )}
    </div>
  );
};