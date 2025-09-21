// UI动画组件库 - v1.2.0
import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { cn } from '@/lib/utils';

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
  }, [currentIndex, text, speed, onComplete]);

  return (
    <span className={className}>
      {displayText}
      {currentIndex < text.length && (
        <motion.span
          animate={{ opacity: [1, 0] }}
          transition={{ duration: 0.5, repeat: Infinity }}
          className="inline-block w-0.5 h-4 bg-current ml-0.5"
        />
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
  amplitude = [],
  className
}) => {
  const bars = 5;
  const [amplitudes, setAmplitudes] = useState<number[]>(new Array(bars).fill(0.2));

  useEffect(() => {
    if (isActive) {
      const interval = setInterval(() => {
        if (amplitude.length > 0) {
          setAmplitudes(amplitude.slice(0, bars));
        } else {
          // 生成随机波形
          setAmplitudes(Array.from({ length: bars }, () => 0.2 + Math.random() * 0.8));
        }
      }, 100);
      return () => clearInterval(interval);
    } else {
      setAmplitudes(new Array(bars).fill(0.2));
    }
  }, [isActive, amplitude]);

  return (
    <div className={cn("flex items-center gap-1 h-8", className)}>
      {amplitudes.map((amp, i) => (
        <motion.div
          key={i}
          className="w-1 bg-gradient-to-t from-blue-500 to-blue-300 rounded-full"
          animate={{
            height: isActive ? `${amp * 100}%` : '20%',
            opacity: isActive ? 1 : 0.3
          }}
          transition={{
            duration: 0.2,
            ease: "easeInOut"
          }}
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
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className={cn("flex gap-1", className)}
        >
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-gray-400 rounded-full"
              animate={{
                y: [0, -8, 0],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{
                duration: 1.2,
                repeat: Infinity,
                delay: i * 0.2,
                ease: "easeInOut"
              }}
            />
          ))}
        </motion.div>
      )}
    </AnimatePresence>
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
    <motion.div
      initial={isNew ? { opacity: 0, y: 20, scale: 0.9 } : false}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "max-w-[70%] p-4 rounded-2xl shadow-sm",
        isUser ? "bg-blue-500 text-white ml-auto" : "bg-gray-100 text-gray-800",
        className
      )}
    >
      {children}
    </motion.div>
  );
};

// ==================== 流光边框 ====================
interface GlowBorderProps {
  children: React.ReactNode;
  isActive?: boolean;
  color?: string;
  className?: string;
}

export const GlowBorder: React.FC<GlowBorderProps> = ({
  children,
  isActive = false,
  color = "blue",
  className
}) => {
  return (
    <div className={cn("relative", className)}>
      {isActive && (
        <motion.div
          className={`absolute inset-0 rounded-lg bg-gradient-to-r from-${color}-400 via-${color}-500 to-${color}-400`}
          animate={{
            backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear"
          }}
          style={{
            backgroundSize: "200% 200%",
            filter: "blur(8px)",
            opacity: 0.5
          }}
        />
      )}
      <div className="relative bg-white rounded-lg">
        {children}
      </div>
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

  return (
    <AnimatePresence>
      {isActive && (
        <div className={cn("relative", className)}>
          <motion.div
            className={cn(
              sizeClasses[size],
              `bg-${color}-500 rounded-full`
            )}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [1, 0.8, 1]
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <motion.div
            className={cn(
              sizeClasses[size],
              `bg-${color}-500 rounded-full absolute top-0 left-0`
            )}
            animate={{
              scale: [1, 2, 2],
              opacity: [0.5, 0, 0]
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeOut"
            }}
          />
        </div>
      )}
    </AnimatePresence>
  );
};

// ==================== 滑动面板 ====================
interface SlidePanel {
  isOpen: boolean;
  direction?: 'left' | 'right' | 'top' | 'bottom';
  children: React.ReactNode;
  onClose?: () => void;
  className?: string;
}

export const SlidePanel: React.FC<SlidePanel> = ({
  isOpen,
  direction = 'right',
  children,
  onClose,
  className
}) => {
  const variants = {
    left: { x: '-100%' },
    right: { x: '100%' },
    top: { y: '-100%' },
    bottom: { y: '100%' }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black z-40"
            onClick={onClose}
          />
          <motion.div
            initial={variants[direction]}
            animate={{ x: 0, y: 0 }}
            exit={variants[direction]}
            transition={{ type: "spring", damping: 20 }}
            className={cn(
              "fixed bg-white shadow-xl z-50",
              direction === 'left' && "left-0 top-0 h-full",
              direction === 'right' && "right-0 top-0 h-full",
              direction === 'top' && "top-0 left-0 w-full",
              direction === 'bottom' && "bottom-0 left-0 w-full",
              className
            )}
          >
            {children}
          </motion.div>
        </>
      )}
    </AnimatePresence>
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
    <motion.div
      className={cn(
        "bg-gray-200",
        variantClasses[variant],
        className
      )}
      style={{ width, height }}
      animate={{
        backgroundImage: [
          'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)',
          'linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%)'
        ]
      }}
      transition={{
        duration: 1.5,
        repeat: Infinity,
        ease: "linear"
      }}
    />
  );
};

// ==================== 渐变文字 ====================
interface GradientTextProps {
  children: React.ReactNode;
  from?: string;
  to?: string;
  animate?: boolean;
  className?: string;
}

export const GradientText: React.FC<GradientTextProps> = ({
  children,
  from = 'blue-500',
  to = 'purple-500',
  animate = false,
  className
}) => {
  return (
    <motion.span
      className={cn(
        "bg-clip-text text-transparent",
        `bg-gradient-to-r from-${from} to-${to}`,
        className
      )}
      animate={animate ? {
        backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
      } : undefined}
      transition={animate ? {
        duration: 5,
        repeat: Infinity,
        ease: "linear"
      } : undefined}
      style={{
        backgroundSize: animate ? "200% 200%" : undefined
      }}
    >
      {children}
    </motion.span>
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
  }, [isVisible, onClose]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.9 }}
          className={cn(
            "fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-3 rounded-lg text-white shadow-lg",
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
        </motion.div>
      )}
    </AnimatePresence>
  );
};
