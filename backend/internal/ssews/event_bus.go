package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"voicehelper/backend/pkg/cache"
	"voicehelper/backend/pkg/types"
)

// EventBus 事件总线接口
type EventBus interface {
	Publish(topic string, event *types.EventEnvelope) error
	Subscribe(topic string, handler func(*types.EventEnvelope)) error
	Unsubscribe(topic string) error
	Close() error
}

// RedisEventBus 基于Redis的事件总线实现
type RedisEventBus struct {
	redisClient *cache.RedisClient
	subscribers map[string][]func(*types.EventEnvelope)
	mutex       sync.RWMutex
	ctx         context.Context
	cancel      context.CancelFunc
}

// NewEventBus 创建事件总线
func NewEventBus(redisClient *cache.RedisClient) EventBus {
	ctx, cancel := context.WithCancel(context.Background())

	bus := &RedisEventBus{
		redisClient: redisClient,
		subscribers: make(map[string][]func(*types.EventEnvelope)),
		ctx:         ctx,
		cancel:      cancel,
	}

	// 启动订阅协程
	go bus.subscriptionWorker()

	return bus
}

// Publish 发布事件
func (r *RedisEventBus) Publish(topic string, event *types.EventEnvelope) error {
	// 序列化事件
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %v", err)
	}

	// 发布到Redis
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	channel := fmt.Sprintf("voicehelper:events:%s", topic)
	if err := r.redisClient.Client.Publish(ctx, channel, data).Err(); err != nil {
		return fmt.Errorf("failed to publish event: %v", err)
	}

	// 同时触发本地订阅者
	r.triggerLocalSubscribers(topic, event)

	logrus.WithFields(logrus.Fields{
		"topic":      topic,
		"event_type": event.Type,
		"session_id": event.Meta.SessionID,
		"trace_id":   event.Meta.TraceID,
	}).Debug("Event published")

	return nil
}

// Subscribe 订阅事件
func (r *RedisEventBus) Subscribe(topic string, handler func(*types.EventEnvelope)) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if r.subscribers[topic] == nil {
		r.subscribers[topic] = make([]func(*types.EventEnvelope), 0)
	}

	r.subscribers[topic] = append(r.subscribers[topic], handler)

	logrus.WithField("topic", topic).Debug("Event subscription added")

	return nil
}

// Unsubscribe 取消订阅
func (r *RedisEventBus) Unsubscribe(topic string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	delete(r.subscribers, topic)

	logrus.WithField("topic", topic).Debug("Event subscription removed")

	return nil
}

// Close 关闭事件总线
func (r *RedisEventBus) Close() error {
	r.cancel()
	return nil
}

// triggerLocalSubscribers 触发本地订阅者
func (r *RedisEventBus) triggerLocalSubscribers(topic string, event *types.EventEnvelope) {
	r.mutex.RLock()
	handlers := r.subscribers[topic]
	r.mutex.RUnlock()

	for _, handler := range handlers {
		go func(h func(*types.EventEnvelope)) {
			defer func() {
				if r := recover(); r != nil {
					logrus.WithFields(logrus.Fields{
						"topic": topic,
						"error": r,
					}).Error("Event handler panicked")
				}
			}()

			h(event)
		}(handler)
	}
}

// subscriptionWorker Redis订阅工作协程
func (r *RedisEventBus) subscriptionWorker() {
	pubsub := r.redisClient.Client.PSubscribe(r.ctx, "voicehelper:events:*")
	defer pubsub.Close()

	ch := pubsub.Channel()

	for {
		select {
		case <-r.ctx.Done():
			return
		case msg := <-ch:
			if msg == nil {
				continue
			}

			// 解析事件
			var event types.EventEnvelope
			if err := json.Unmarshal([]byte(msg.Payload), &event); err != nil {
				logrus.WithError(err).Error("Failed to unmarshal event")
				continue
			}

			// 提取topic
			topic := msg.Channel[len("voicehelper:events:"):]

			// 触发本地订阅者
			r.triggerLocalSubscribers(topic, &event)
		}
	}
}

// InMemoryEventBus 内存事件总线实现（用于测试）
type InMemoryEventBus struct {
	subscribers map[string][]func(*types.EventEnvelope)
	mutex       sync.RWMutex
}

// NewInMemoryEventBus 创建内存事件总线
func NewInMemoryEventBus() EventBus {
	return &InMemoryEventBus{
		subscribers: make(map[string][]func(*types.EventEnvelope)),
	}
}

// Publish 发布事件
func (i *InMemoryEventBus) Publish(topic string, event *types.EventEnvelope) error {
	i.mutex.RLock()
	handlers := i.subscribers[topic]
	i.mutex.RUnlock()

	for _, handler := range handlers {
		go func(h func(*types.EventEnvelope)) {
			defer func() {
				if r := recover(); r != nil {
					logrus.WithFields(logrus.Fields{
						"topic": topic,
						"error": r,
					}).Error("Event handler panicked")
				}
			}()

			h(event)
		}(handler)
	}

	return nil
}

// Subscribe 订阅事件
func (i *InMemoryEventBus) Subscribe(topic string, handler func(*types.EventEnvelope)) error {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	if i.subscribers[topic] == nil {
		i.subscribers[topic] = make([]func(*types.EventEnvelope), 0)
	}

	i.subscribers[topic] = append(i.subscribers[topic], handler)

	return nil
}

// Unsubscribe 取消订阅
func (i *InMemoryEventBus) Unsubscribe(topic string) error {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	delete(i.subscribers, topic)

	return nil
}

// Close 关闭事件总线
func (i *InMemoryEventBus) Close() error {
	return nil
}
