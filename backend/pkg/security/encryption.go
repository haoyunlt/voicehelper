package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// EncryptionManager handles data encryption and key management
type EncryptionManager struct {
	logger    *logrus.Logger
	keyStore  map[string]*EncryptionKey
	mu        sync.RWMutex
	config    *EncryptionConfig
	masterKey []byte
}

// EncryptionConfig contains encryption configuration
type EncryptionConfig struct {
	Algorithm           string        `json:"algorithm"`
	KeySize             int           `json:"key_size"`
	KeyRotationInterval time.Duration `json:"key_rotation_interval"`
	EnableKeyRotation   bool          `json:"enable_key_rotation"`
	MasterKeyPath       string        `json:"master_key_path"`
	HSMEnabled          bool          `json:"hsm_enabled"`
	HSMConfig           *HSMConfig    `json:"hsm_config,omitempty"`
}

// HSMConfig contains Hardware Security Module configuration
type HSMConfig struct {
	Provider    string            `json:"provider"`
	Endpoint    string            `json:"endpoint"`
	Credentials map[string]string `json:"credentials"`
	KeyLabel    string            `json:"key_label"`
}

// EncryptionKey represents an encryption key with metadata
type EncryptionKey struct {
	ID        string                 `json:"id"`
	Algorithm string                 `json:"algorithm"`
	KeyData   []byte                 `json:"-"` // Never serialize key data
	Purpose   KeyPurpose             `json:"purpose"`
	Status    KeyStatus              `json:"status"`
	CreatedAt time.Time              `json:"created_at"`
	ExpiresAt time.Time              `json:"expires_at"`
	RotatedAt *time.Time             `json:"rotated_at,omitempty"`
	Usage     *KeyUsage              `json:"usage"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// KeyPurpose represents the purpose of an encryption key
type KeyPurpose string

const (
	KeyPurposeDataEncryption    KeyPurpose = "data_encryption"
	KeyPurposeTokenSigning      KeyPurpose = "token_signing"
	KeyPurposePasswordHashing   KeyPurpose = "password_hashing"
	KeyPurposeSessionEncryption KeyPurpose = "session_encryption"
	KeyPurposeFileEncryption    KeyPurpose = "file_encryption"
)

// KeyStatus represents the status of an encryption key
type KeyStatus string

const (
	KeyStatusActive     KeyStatus = "active"
	KeyStatusRotating   KeyStatus = "rotating"
	KeyStatusDeprecated KeyStatus = "deprecated"
	KeyStatusRevoked    KeyStatus = "revoked"
)

// KeyUsage tracks key usage statistics
type KeyUsage struct {
	EncryptionCount int       `json:"encryption_count"`
	DecryptionCount int       `json:"decryption_count"`
	LastUsed        time.Time `json:"last_used"`
	BytesEncrypted  int64     `json:"bytes_encrypted"`
	BytesDecrypted  int64     `json:"bytes_decrypted"`
}

// EncryptedData represents encrypted data with metadata
type EncryptedData struct {
	KeyID      string                 `json:"key_id"`
	Algorithm  string                 `json:"algorithm"`
	Ciphertext string                 `json:"ciphertext"` // Base64 encoded
	IV         string                 `json:"iv"`         // Base64 encoded initialization vector
	Tag        string                 `json:"tag"`        // Base64 encoded authentication tag (for AEAD)
	Timestamp  time.Time              `json:"timestamp"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// NewEncryptionManager creates a new encryption manager
func NewEncryptionManager() *EncryptionManager {
	config := &EncryptionConfig{
		Algorithm:           "AES-256-GCM",
		KeySize:             32,                  // 256 bits
		KeyRotationInterval: 30 * 24 * time.Hour, // 30 days
		EnableKeyRotation:   true,
		MasterKeyPath:       "/etc/chatbot/master.key",
		HSMEnabled:          false,
	}

	manager := &EncryptionManager{
		logger:   logrus.New(),
		keyStore: make(map[string]*EncryptionKey),
		config:   config,
	}

	// Initialize master key
	if err := manager.initializeMasterKey(); err != nil {
		manager.logger.Errorf("Failed to initialize master key: %v", err)
	}

	// Start key rotation routine
	if config.EnableKeyRotation {
		go manager.keyRotationRoutine()
	}

	return manager
}

// initializeMasterKey initializes or loads the master encryption key
func (em *EncryptionManager) initializeMasterKey() error {
	// In production, this would load from a secure key management system
	// For now, generate a random master key
	masterKey := make([]byte, 32)
	if _, err := rand.Read(masterKey); err != nil {
		return fmt.Errorf("failed to generate master key: %v", err)
	}

	em.masterKey = masterKey
	em.logger.Info("Master encryption key initialized")

	return nil
}

// GenerateKey generates a new encryption key
func (em *EncryptionManager) GenerateKey(purpose KeyPurpose, expirationDays int) (*EncryptionKey, error) {
	keyData := make([]byte, em.config.KeySize)
	if _, err := rand.Read(keyData); err != nil {
		return nil, fmt.Errorf("failed to generate key data: %v", err)
	}

	key := &EncryptionKey{
		ID:        generateKeyID(),
		Algorithm: em.config.Algorithm,
		KeyData:   keyData,
		Purpose:   purpose,
		Status:    KeyStatusActive,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().AddDate(0, 0, expirationDays),
		Usage: &KeyUsage{
			EncryptionCount: 0,
			DecryptionCount: 0,
			BytesEncrypted:  0,
			BytesDecrypted:  0,
		},
		Metadata: make(map[string]interface{}),
	}

	em.mu.Lock()
	em.keyStore[key.ID] = key
	em.mu.Unlock()

	em.logger.Infof("Generated new encryption key: %s (purpose: %s)", key.ID, purpose)

	return key, nil
}

// GetKey retrieves an encryption key by ID
func (em *EncryptionManager) GetKey(keyID string) (*EncryptionKey, error) {
	em.mu.RLock()
	defer em.mu.RUnlock()

	key, exists := em.keyStore[keyID]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	if key.Status == KeyStatusRevoked {
		return nil, fmt.Errorf("key is revoked: %s", keyID)
	}

	if time.Now().After(key.ExpiresAt) {
		return nil, fmt.Errorf("key is expired: %s", keyID)
	}

	return key, nil
}

// Encrypt encrypts data using the specified key
func (em *EncryptionManager) Encrypt(keyID string, plaintext []byte) (*EncryptedData, error) {
	key, err := em.GetKey(keyID)
	if err != nil {
		return nil, err
	}

	switch key.Algorithm {
	case "AES-256-GCM":
		return em.encryptAESGCM(key, plaintext)
	default:
		return nil, fmt.Errorf("unsupported algorithm: %s", key.Algorithm)
	}
}

// Decrypt decrypts data using the specified key
func (em *EncryptionManager) Decrypt(encryptedData *EncryptedData) ([]byte, error) {
	key, err := em.GetKey(encryptedData.KeyID)
	if err != nil {
		return nil, err
	}

	switch encryptedData.Algorithm {
	case "AES-256-GCM":
		return em.decryptAESGCM(key, encryptedData)
	default:
		return nil, fmt.Errorf("unsupported algorithm: %s", encryptedData.Algorithm)
	}
}

// encryptAESGCM encrypts data using AES-256-GCM
func (em *EncryptionManager) encryptAESGCM(key *EncryptionKey, plaintext []byte) (*EncryptedData, error) {
	block, err := aes.NewCipher(key.KeyData)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %v", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %v", err)
	}

	// Generate random IV
	iv := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, iv); err != nil {
		return nil, fmt.Errorf("failed to generate IV: %v", err)
	}

	// Encrypt and authenticate
	ciphertext := gcm.Seal(nil, iv, plaintext, nil)

	// Update key usage
	em.mu.Lock()
	key.Usage.EncryptionCount++
	key.Usage.BytesEncrypted += int64(len(plaintext))
	key.Usage.LastUsed = time.Now()
	em.mu.Unlock()

	return &EncryptedData{
		KeyID:      key.ID,
		Algorithm:  key.Algorithm,
		Ciphertext: base64.StdEncoding.EncodeToString(ciphertext),
		IV:         base64.StdEncoding.EncodeToString(iv),
		Timestamp:  time.Now(),
	}, nil
}

// decryptAESGCM decrypts data using AES-256-GCM
func (em *EncryptionManager) decryptAESGCM(key *EncryptionKey, encryptedData *EncryptedData) ([]byte, error) {
	block, err := aes.NewCipher(key.KeyData)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %v", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %v", err)
	}

	// Decode base64 data
	ciphertext, err := base64.StdEncoding.DecodeString(encryptedData.Ciphertext)
	if err != nil {
		return nil, fmt.Errorf("failed to decode ciphertext: %v", err)
	}

	iv, err := base64.StdEncoding.DecodeString(encryptedData.IV)
	if err != nil {
		return nil, fmt.Errorf("failed to decode IV: %v", err)
	}

	// Decrypt and verify
	plaintext, err := gcm.Open(nil, iv, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %v", err)
	}

	// Update key usage
	em.mu.Lock()
	key.Usage.DecryptionCount++
	key.Usage.BytesDecrypted += int64(len(plaintext))
	key.Usage.LastUsed = time.Now()
	em.mu.Unlock()

	return plaintext, nil
}

// EncryptString encrypts a string and returns base64 encoded result
func (em *EncryptionManager) EncryptString(keyID, plaintext string) (string, error) {
	encryptedData, err := em.Encrypt(keyID, []byte(plaintext))
	if err != nil {
		return "", err
	}

	// Serialize encrypted data to JSON and encode as base64
	jsonData := fmt.Sprintf(`{"key_id":"%s","algorithm":"%s","ciphertext":"%s","iv":"%s","timestamp":"%s"}`,
		encryptedData.KeyID,
		encryptedData.Algorithm,
		encryptedData.Ciphertext,
		encryptedData.IV,
		encryptedData.Timestamp.Format(time.RFC3339))

	return base64.StdEncoding.EncodeToString([]byte(jsonData)), nil
}

// DecryptString decrypts a base64 encoded encrypted string
func (em *EncryptionManager) DecryptString(encryptedString string) (string, error) {
	// Decode base64
	_, err := base64.StdEncoding.DecodeString(encryptedString)
	if err != nil {
		return "", fmt.Errorf("failed to decode encrypted string: %v", err)
	}

	// Parse JSON (simplified parsing for this example)
	// In production, use proper JSON parsing
	var encryptedData EncryptedData
	// This is a simplified implementation - in production, use json.Unmarshal

	plaintext, err := em.Decrypt(&encryptedData)
	if err != nil {
		return "", err
	}

	return string(plaintext), nil
}

// RotateKey rotates an encryption key
func (em *EncryptionManager) RotateKey(keyID string) (*EncryptionKey, error) {
	em.mu.Lock()
	defer em.mu.Unlock()

	oldKey, exists := em.keyStore[keyID]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	// Mark old key as deprecated
	oldKey.Status = KeyStatusDeprecated
	now := time.Now()
	oldKey.RotatedAt = &now

	// Generate new key with same purpose
	newKeyData := make([]byte, em.config.KeySize)
	if _, err := rand.Read(newKeyData); err != nil {
		return nil, fmt.Errorf("failed to generate new key data: %v", err)
	}

	newKey := &EncryptionKey{
		ID:        generateKeyID(),
		Algorithm: oldKey.Algorithm,
		KeyData:   newKeyData,
		Purpose:   oldKey.Purpose,
		Status:    KeyStatusActive,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(em.config.KeyRotationInterval),
		Usage: &KeyUsage{
			EncryptionCount: 0,
			DecryptionCount: 0,
			BytesEncrypted:  0,
			BytesDecrypted:  0,
		},
		Metadata: make(map[string]interface{}),
	}

	em.keyStore[newKey.ID] = newKey

	em.logger.Infof("Rotated encryption key: %s -> %s", keyID, newKey.ID)

	return newKey, nil
}

// RevokeKey revokes an encryption key
func (em *EncryptionManager) RevokeKey(keyID string, reason string) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	key, exists := em.keyStore[keyID]
	if !exists {
		return fmt.Errorf("key not found: %s", keyID)
	}

	key.Status = KeyStatusRevoked
	key.Metadata["revocation_reason"] = reason
	key.Metadata["revoked_at"] = time.Now()

	em.logger.Warnf("Revoked encryption key: %s (reason: %s)", keyID, reason)

	return nil
}

// ListKeys returns all encryption keys with their metadata
func (em *EncryptionManager) ListKeys() []*EncryptionKey {
	em.mu.RLock()
	defer em.mu.RUnlock()

	keys := make([]*EncryptionKey, 0, len(em.keyStore))
	for _, key := range em.keyStore {
		// Create a copy without the key data for security
		keyCopy := &EncryptionKey{
			ID:        key.ID,
			Algorithm: key.Algorithm,
			Purpose:   key.Purpose,
			Status:    key.Status,
			CreatedAt: key.CreatedAt,
			ExpiresAt: key.ExpiresAt,
			RotatedAt: key.RotatedAt,
			Usage:     key.Usage,
			Metadata:  key.Metadata,
		}
		keys = append(keys, keyCopy)
	}

	return keys
}

// GetKeysByPurpose returns all keys for a specific purpose
func (em *EncryptionManager) GetKeysByPurpose(purpose KeyPurpose) []*EncryptionKey {
	em.mu.RLock()
	defer em.mu.RUnlock()

	var keys []*EncryptionKey
	for _, key := range em.keyStore {
		if key.Purpose == purpose && key.Status == KeyStatusActive {
			keyCopy := &EncryptionKey{
				ID:        key.ID,
				Algorithm: key.Algorithm,
				Purpose:   key.Purpose,
				Status:    key.Status,
				CreatedAt: key.CreatedAt,
				ExpiresAt: key.ExpiresAt,
				RotatedAt: key.RotatedAt,
				Usage:     key.Usage,
				Metadata:  key.Metadata,
			}
			keys = append(keys, keyCopy)
		}
	}

	return keys
}

// keyRotationRoutine runs in the background to rotate keys
func (em *EncryptionManager) keyRotationRoutine() {
	ticker := time.NewTicker(24 * time.Hour) // Check daily
	defer ticker.Stop()

	for range ticker.C {
		em.mu.RLock()
		keysToRotate := make([]string, 0)

		for keyID, key := range em.keyStore {
			if key.Status == KeyStatusActive &&
				time.Now().After(key.CreatedAt.Add(em.config.KeyRotationInterval)) {
				keysToRotate = append(keysToRotate, keyID)
			}
		}
		em.mu.RUnlock()

		for _, keyID := range keysToRotate {
			if _, err := em.RotateKey(keyID); err != nil {
				em.logger.Errorf("Failed to rotate key %s: %v", keyID, err)
			}
		}
	}
}

// RSA key pair generation and operations

// GenerateRSAKeyPair generates an RSA key pair
func (em *EncryptionManager) GenerateRSAKeyPair(keySize int) (*rsa.PrivateKey, *rsa.PublicKey, error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, keySize)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate RSA key pair: %v", err)
	}

	return privateKey, &privateKey.PublicKey, nil
}

// EncryptRSA encrypts data using RSA public key
func (em *EncryptionManager) EncryptRSA(publicKey *rsa.PublicKey, plaintext []byte) ([]byte, error) {
	ciphertext, err := rsa.EncryptOAEP(sha256.New(), rand.Reader, publicKey, plaintext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt with RSA: %v", err)
	}

	return ciphertext, nil
}

// DecryptRSA decrypts data using RSA private key
func (em *EncryptionManager) DecryptRSA(privateKey *rsa.PrivateKey, ciphertext []byte) ([]byte, error) {
	plaintext, err := rsa.DecryptOAEP(sha256.New(), rand.Reader, privateKey, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt with RSA: %v", err)
	}

	return plaintext, nil
}

// ExportPublicKeyPEM exports RSA public key as PEM
func (em *EncryptionManager) ExportPublicKeyPEM(publicKey *rsa.PublicKey) (string, error) {
	pubKeyBytes, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return "", fmt.Errorf("failed to marshal public key: %v", err)
	}

	pubKeyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: pubKeyBytes,
	})

	return string(pubKeyPEM), nil
}

// ImportPublicKeyPEM imports RSA public key from PEM
func (em *EncryptionManager) ImportPublicKeyPEM(pemData string) (*rsa.PublicKey, error) {
	block, _ := pem.Decode([]byte(pemData))
	if block == nil {
		return nil, fmt.Errorf("failed to decode PEM block")
	}

	pubKey, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse public key: %v", err)
	}

	rsaPubKey, ok := pubKey.(*rsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("not an RSA public key")
	}

	return rsaPubKey, nil
}

// Utility functions

func generateKeyID() string {
	timestamp := time.Now().UnixNano()
	return fmt.Sprintf("key_%d", timestamp)
}

// GetKeyUsageStats returns usage statistics for all keys
func (em *EncryptionManager) GetKeyUsageStats() map[string]*KeyUsage {
	em.mu.RLock()
	defer em.mu.RUnlock()

	stats := make(map[string]*KeyUsage)
	for keyID, key := range em.keyStore {
		stats[keyID] = &KeyUsage{
			EncryptionCount: key.Usage.EncryptionCount,
			DecryptionCount: key.Usage.DecryptionCount,
			LastUsed:        key.Usage.LastUsed,
			BytesEncrypted:  key.Usage.BytesEncrypted,
			BytesDecrypted:  key.Usage.BytesDecrypted,
		}
	}

	return stats
}

// CleanupExpiredKeys removes expired and revoked keys
func (em *EncryptionManager) CleanupExpiredKeys() int {
	em.mu.Lock()
	defer em.mu.Unlock()

	cleaned := 0
	for keyID, key := range em.keyStore {
		if (key.Status == KeyStatusRevoked &&
			key.Metadata["revoked_at"] != nil &&
			time.Since(key.Metadata["revoked_at"].(time.Time)) > 90*24*time.Hour) ||
			(key.Status == KeyStatusDeprecated &&
				key.RotatedAt != nil &&
				time.Since(*key.RotatedAt) > 90*24*time.Hour) {
			delete(em.keyStore, keyID)
			cleaned++
			em.logger.Infof("Cleaned up expired key: %s", keyID)
		}
	}

	return cleaned
}
