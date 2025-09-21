package service

import (
	"context"
	"fmt"
)

// ChatService 对话服务
type ChatService struct {
	algoService *AlgoService
}

func NewChatService(algoService *AlgoService) *ChatService {
	return &ChatService{
		algoService: algoService,
	}
}

// ChatRequest 对话请求
type ChatRequest struct {
	ConversationID string    `json:"conversation_id,omitempty"`
	Messages       []Message `json:"messages"`
	TopK           int       `json:"top_k,omitempty"`
	Temperature    float64   `json:"temperature,omitempty"`
}

// StreamChat 流式对话
func (s *ChatService) StreamChat(ctx context.Context, req *ChatRequest) (<-chan *QueryResponse, error) {
	// 设置默认值
	if req.TopK == 0 {
		req.TopK = 5
	}
	if req.Temperature == 0 {
		req.Temperature = 0.3
	}

	// 构建算法服务请求
	algoReq := &QueryRequest{
		Messages:    req.Messages,
		TopK:        req.TopK,
		Temperature: req.Temperature,
		MaxTokens:   1024,
	}

	// 调用算法服务
	responseCh, err := s.algoService.Query(ctx, algoReq)
	if err != nil {
		return nil, fmt.Errorf("query algo service: %w", err)
	}

	return responseCh, nil
}
