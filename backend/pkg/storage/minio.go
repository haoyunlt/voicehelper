package storage

import (
	"context"
	"fmt"
	"io"
	"mime"
	"path/filepath"
	"strings"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/sirupsen/logrus"
)

// Config MinIO配置
type Config struct {
	Endpoint        string `json:"endpoint" yaml:"endpoint"`
	AccessKeyID     string `json:"access_key_id" yaml:"access_key_id"`
	SecretAccessKey string `json:"secret_access_key" yaml:"secret_access_key"`
	UseSSL          bool   `json:"use_ssl" yaml:"use_ssl"`
	BucketName      string `json:"bucket_name" yaml:"bucket_name"`
	Region          string `json:"region" yaml:"region"`
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		Endpoint:        "localhost:9000",
		AccessKeyID:     "minioadmin",
		SecretAccessKey: "minioadmin",
		UseSSL:          false,
		BucketName:      "chatbot",
		Region:          "us-east-1",
	}
}

// ObjectInfo 对象信息
type ObjectInfo struct {
	Key          string            `json:"key"`
	Size         int64             `json:"size"`
	ContentType  string            `json:"content_type"`
	ETag         string            `json:"etag"`
	LastModified time.Time         `json:"last_modified"`
	Metadata     map[string]string `json:"metadata"`
}

// UploadResult 上传结果
type UploadResult struct {
	Key         string `json:"key"`
	Bucket      string `json:"bucket"`
	Size        int64  `json:"size"`
	ETag        string `json:"etag"`
	ContentType string `json:"content_type"`
	URL         string `json:"url"`
}

// ObjectStorage 对象存储接口
type ObjectStorage interface {
	// 基础操作
	Upload(ctx context.Context, key string, reader io.Reader, size int64, contentType string) (*UploadResult, error)
	Download(ctx context.Context, key string) (io.ReadCloser, error)
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)

	// 批量操作
	ListObjects(ctx context.Context, prefix string, recursive bool) ([]*ObjectInfo, error)
	DeleteMultiple(ctx context.Context, keys []string) error

	// URL生成
	GetPresignedURL(ctx context.Context, key string, expires time.Duration) (string, error)
	GetPresignedPutURL(ctx context.Context, key string, expires time.Duration) (string, error)

	// 元数据
	GetObjectInfo(ctx context.Context, key string) (*ObjectInfo, error)
	UpdateMetadata(ctx context.Context, key string, metadata map[string]string) error

	// Bucket操作
	CreateBucket(ctx context.Context, bucketName string) error
	ListBuckets(ctx context.Context) ([]string, error)
	DeleteBucket(ctx context.Context, bucketName string) error
}

// MinIOStorage MinIO存储实现
type MinIOStorage struct {
	client     *minio.Client
	config     *Config
	bucketName string
}

// NewMinIOStorage 创建MinIO存储
func NewMinIOStorage(config *Config) (*MinIOStorage, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// 创建MinIO客户端
	client, err := minio.New(config.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(config.AccessKeyID, config.SecretAccessKey, ""),
		Secure: config.UseSSL,
		Region: config.Region,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create MinIO client: %w", err)
	}

	storage := &MinIOStorage{
		client:     client,
		config:     config,
		bucketName: config.BucketName,
	}

	// 确保默认bucket存在
	ctx := context.Background()
	exists, err := client.BucketExists(ctx, config.BucketName)
	if err != nil {
		return nil, fmt.Errorf("failed to check bucket existence: %w", err)
	}

	if !exists {
		err = client.MakeBucket(ctx, config.BucketName, minio.MakeBucketOptions{
			Region: config.Region,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create bucket: %w", err)
		}
		logrus.Infof("Created MinIO bucket: %s", config.BucketName)
	}

	logrus.Infof("Connected to MinIO: %s (bucket: %s)", config.Endpoint, config.BucketName)
	return storage, nil
}

// Upload 上传对象
func (s *MinIOStorage) Upload(ctx context.Context, key string, reader io.Reader, size int64, contentType string) (*UploadResult, error) {
	// 自动检测content type
	if contentType == "" {
		contentType = mime.TypeByExtension(filepath.Ext(key))
		if contentType == "" {
			contentType = "application/octet-stream"
		}
	}

	// 上传选项
	opts := minio.PutObjectOptions{
		ContentType: contentType,
		UserMetadata: map[string]string{
			"uploaded_at": time.Now().UTC().Format(time.RFC3339),
		},
	}

	// 执行上传
	info, err := s.client.PutObject(ctx, s.bucketName, key, reader, size, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to upload object: %w", err)
	}

	// 生成访问URL
	url := fmt.Sprintf("%s/%s/%s", s.getEndpointURL(), s.bucketName, key)

	result := &UploadResult{
		Key:         key,
		Bucket:      s.bucketName,
		Size:        info.Size,
		ETag:        info.ETag,
		ContentType: contentType,
		URL:         url,
	}

	logrus.Debugf("Uploaded object: %s (size: %d, etag: %s)", key, info.Size, info.ETag)
	return result, nil
}

// Download 下载对象
func (s *MinIOStorage) Download(ctx context.Context, key string) (io.ReadCloser, error) {
	object, err := s.client.GetObject(ctx, s.bucketName, key, minio.GetObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get object: %w", err)
	}

	// 验证对象是否存在
	_, err = object.Stat()
	if err != nil {
		object.Close()
		return nil, fmt.Errorf("failed to stat object: %w", err)
	}

	return object, nil
}

// Delete 删除对象
func (s *MinIOStorage) Delete(ctx context.Context, key string) error {
	err := s.client.RemoveObject(ctx, s.bucketName, key, minio.RemoveObjectOptions{})
	if err != nil {
		return fmt.Errorf("failed to delete object: %w", err)
	}

	logrus.Debugf("Deleted object: %s", key)
	return nil
}

// Exists 检查对象是否存在
func (s *MinIOStorage) Exists(ctx context.Context, key string) (bool, error) {
	_, err := s.client.StatObject(ctx, s.bucketName, key, minio.StatObjectOptions{})
	if err != nil {
		errResp := minio.ToErrorResponse(err)
		if errResp.Code == "NoSuchKey" {
			return false, nil
		}
		return false, fmt.Errorf("failed to stat object: %w", err)
	}
	return true, nil
}

// ListObjects 列出对象
func (s *MinIOStorage) ListObjects(ctx context.Context, prefix string, recursive bool) ([]*ObjectInfo, error) {
	opts := minio.ListObjectsOptions{
		Prefix:    prefix,
		Recursive: recursive,
	}

	var objects []*ObjectInfo
	for object := range s.client.ListObjects(ctx, s.bucketName, opts) {
		if object.Err != nil {
			return nil, fmt.Errorf("failed to list objects: %w", object.Err)
		}

		objects = append(objects, &ObjectInfo{
			Key:          object.Key,
			Size:         object.Size,
			ContentType:  object.ContentType,
			ETag:         object.ETag,
			LastModified: object.LastModified,
			Metadata:     object.UserMetadata,
		})
	}

	return objects, nil
}

// DeleteMultiple 批量删除对象
func (s *MinIOStorage) DeleteMultiple(ctx context.Context, keys []string) error {
	objectsCh := make(chan minio.ObjectInfo)

	// 发送要删除的对象
	go func() {
		defer close(objectsCh)
		for _, key := range keys {
			objectsCh <- minio.ObjectInfo{
				Key: key,
			}
		}
	}()

	// 执行批量删除
	for err := range s.client.RemoveObjects(ctx, s.bucketName, objectsCh, minio.RemoveObjectsOptions{}) {
		if err.Err != nil {
			return fmt.Errorf("failed to delete object %s: %w", err.ObjectName, err.Err)
		}
	}

	logrus.Debugf("Deleted %d objects", len(keys))
	return nil
}

// GetPresignedURL 获取预签名下载URL
func (s *MinIOStorage) GetPresignedURL(ctx context.Context, key string, expires time.Duration) (string, error) {
	if expires == 0 {
		expires = 7 * 24 * time.Hour // 默认7天
	}

	url, err := s.client.PresignedGetObject(ctx, s.bucketName, key, expires, nil)
	if err != nil {
		return "", fmt.Errorf("failed to generate presigned URL: %w", err)
	}

	return url.String(), nil
}

// GetPresignedPutURL 获取预签名上传URL
func (s *MinIOStorage) GetPresignedPutURL(ctx context.Context, key string, expires time.Duration) (string, error) {
	if expires == 0 {
		expires = time.Hour // 默认1小时
	}

	url, err := s.client.PresignedPutObject(ctx, s.bucketName, key, expires)
	if err != nil {
		return "", fmt.Errorf("failed to generate presigned put URL: %w", err)
	}

	return url.String(), nil
}

// GetObjectInfo 获取对象信息
func (s *MinIOStorage) GetObjectInfo(ctx context.Context, key string) (*ObjectInfo, error) {
	stat, err := s.client.StatObject(ctx, s.bucketName, key, minio.StatObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to stat object: %w", err)
	}

	return &ObjectInfo{
		Key:          stat.Key,
		Size:         stat.Size,
		ContentType:  stat.ContentType,
		ETag:         stat.ETag,
		LastModified: stat.LastModified,
		Metadata:     stat.UserMetadata,
	}, nil
}

// UpdateMetadata 更新对象元数据
func (s *MinIOStorage) UpdateMetadata(ctx context.Context, key string, metadata map[string]string) error {
	// MinIO不支持直接更新元数据，需要复制对象
	src := minio.CopySrcOptions{
		Bucket: s.bucketName,
		Object: key,
	}

	dst := minio.CopyDestOptions{
		Bucket:          s.bucketName,
		Object:          key,
		UserMetadata:    metadata,
		ReplaceMetadata: true,
	}

	_, err := s.client.CopyObject(ctx, dst, src)
	if err != nil {
		return fmt.Errorf("failed to update metadata: %w", err)
	}

	return nil
}

// CreateBucket 创建bucket
func (s *MinIOStorage) CreateBucket(ctx context.Context, bucketName string) error {
	err := s.client.MakeBucket(ctx, bucketName, minio.MakeBucketOptions{
		Region: s.config.Region,
	})
	if err != nil {
		return fmt.Errorf("failed to create bucket: %w", err)
	}

	logrus.Infof("Created bucket: %s", bucketName)
	return nil
}

// ListBuckets 列出所有bucket
func (s *MinIOStorage) ListBuckets(ctx context.Context) ([]string, error) {
	buckets, err := s.client.ListBuckets(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list buckets: %w", err)
	}

	var names []string
	for _, bucket := range buckets {
		names = append(names, bucket.Name)
	}

	return names, nil
}

// DeleteBucket 删除bucket
func (s *MinIOStorage) DeleteBucket(ctx context.Context, bucketName string) error {
	err := s.client.RemoveBucket(ctx, bucketName)
	if err != nil {
		return fmt.Errorf("failed to delete bucket: %w", err)
	}

	logrus.Infof("Deleted bucket: %s", bucketName)
	return nil
}

// getEndpointURL 获取endpoint URL
func (s *MinIOStorage) getEndpointURL() string {
	protocol := "http"
	if s.config.UseSSL {
		protocol = "https"
	}
	return fmt.Sprintf("%s://%s", protocol, s.config.Endpoint)
}

// GenerateObjectKey 生成对象键
func GenerateObjectKey(prefix, filename string) string {
	// 生成日期路径
	now := time.Now()
	datePath := now.Format("2006/01/02")

	// 生成唯一ID
	timestamp := now.UnixNano()

	// 获取文件扩展名
	ext := filepath.Ext(filename)
	nameWithoutExt := strings.TrimSuffix(filename, ext)

	// 清理文件名（移除特殊字符）
	cleanName := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, nameWithoutExt)

	// 组合最终的key
	return fmt.Sprintf("%s/%s/%s_%d%s", prefix, datePath, cleanName, timestamp, ext)
}
