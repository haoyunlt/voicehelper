package ssews

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type SSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	closed  bool
}

func NewSSEWriter(w http.ResponseWriter) *SSEWriter {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")

	return &SSEWriter{w: w, flusher: flusher}
}

func (s *SSEWriter) WriteEvent(event string, payload interface{}) error {
	if s.closed {
		return fmt.Errorf("writer closed")
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	fmt.Fprintf(s.w, "event: %s\n", event)
	fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.flusher.Flush()

	return nil
}

func (s *SSEWriter) WriteError(code, message string) error {
	return s.WriteEvent("error", ErrorInfo{Code: code, Message: message})
}

func (s *SSEWriter) Close() error {
	s.closed = true
	return nil
}
