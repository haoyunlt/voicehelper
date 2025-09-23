package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/sirupsen/logrus"
)

type Services struct {
	AlgoService *AlgoService
	ChatService *ChatService
}

func NewServices() *Services {
	algoService := NewAlgoService()
	chatService := NewChatService(algoService)

	return &Services{
		AlgoService: algoService,
		ChatService: chatService,
	}
}

// AlgoService 算法服务客户端
type AlgoService struct {
	baseURL    string
	httpClient *http.Client
}

func NewAlgoService() *AlgoService {
	baseURL := os.Getenv("ALGO_SERVICE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}

	return &AlgoService{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// QueryRequest 查询请求
type QueryRequest struct {
	Messages    []Message `json:"messages"`
	TopK        int       `json:"top_k,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// Message 消息结构
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// QueryResponse 查询响应
type QueryResponse struct {
	Type    string      `json:"type"`
	Content string      `json:"content"`
	Refs    []Reference `json:"refs,omitempty"`
}

// Reference 引用信息
type Reference struct {
	ChunkID string  `json:"chunk_id"`
	Source  string  `json:"source"`
	Score   float64 `json:"score"`
}

// Query 调用算法服务查询
func (s *AlgoService) Query(ctx context.Context, req *QueryRequest) (<-chan *QueryResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/query", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	ch := make(chan *QueryResponse, 10)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		decoder := json.NewDecoder(resp.Body)
		for {
			var response QueryResponse
			if err := decoder.Decode(&response); err != nil {
				if err == io.EOF {
					break
				}
				logrus.WithError(err).Error("Failed to decode response")
				break
			}

			select {
			case ch <- &response:
			case <-ctx.Done():
				return
			}
		}
	}()

	return ch, nil
}

// IngestRequest 入库请求
type IngestRequest struct {
	DatasetID string   `json:"dataset_id"`
	Files     []string `json:"files,omitempty"`
	URLs      []string `json:"urls,omitempty"`
	ChunkSize int      `json:"chunk_size,omitempty"`
}

// IngestResponse 入库响应
type IngestResponse struct {
	TaskID string `json:"task_id"`
}

// VoiceQueryRequest 语音查询请求
type VoiceQueryRequest struct {
	ConversationID string `json:"conversation_id"`
	AudioChunk     string `json:"audio_chunk"`
	Seq            int    `json:"seq"`
	Codec          string `json:"codec"`
	SampleRate     int    `json:"sample_rate"`
}

// VoiceQueryResponse 语音查询响应
type VoiceQueryResponse struct {
	Type  string      `json:"type"`
	Seq   int         `json:"seq,omitempty"`
	Text  string      `json:"text,omitempty"`
	PCM   string      `json:"pcm,omitempty"`
	Refs  []Reference `json:"refs,omitempty"`
	Error string      `json:"error,omitempty"`
}

// Ingest 调用算法服务入库
func (s *AlgoService) Ingest(ctx context.Context, req *IngestRequest) (*IngestResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/ingest", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var response IngestResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &response, nil
}

// VoiceQuery 调用算法服务语音查询
func (s *AlgoService) VoiceQuery(ctx context.Context, req *VoiceQueryRequest) (<-chan *VoiceQueryResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/voice/query", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	ch := make(chan *VoiceQueryResponse, 10)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		decoder := json.NewDecoder(resp.Body)
		for {
			var response VoiceQueryResponse
			if err := decoder.Decode(&response); err != nil {
				if err == io.EOF {
					break
				}
				logrus.WithError(err).Error("Failed to decode voice response")
				break
			}

			select {
			case ch <- &response:
			case <-ctx.Done():
				return
			}
		}
	}()

	return ch, nil
}

// CancelRequest 取消请求
func (s *AlgoService) CancelRequest(ctx context.Context, requestID string) error {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/cancel", nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("X-Request-ID", requestID)

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}
