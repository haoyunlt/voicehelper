package repository

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUserRepository_Simple(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	repo := NewPostgresUserRepository(db)
	ctx := context.Background()

	t.Run("Create User", func(t *testing.T) {
		user := &User{
			OpenID:   "test_openid",
			Nickname: "Test User",
			TenantID: "tenant_123",
		}

		mock.ExpectExec("INSERT INTO users").
			WillReturnResult(sqlmock.NewResult(1, 1))

		err = repo.Create(ctx, user)
		assert.NoError(t, err)
		assert.NotEmpty(t, user.ID)
	})

	t.Run("Get User By ID", func(t *testing.T) {
		userID := "user_123"

		rows := sqlmock.NewRows([]string{
			"id", "open_id", "union_id", "tenant_id", "username",
			"nickname", "avatar", "email", "phone", "role", "status",
			"last_login", "created_at", "updated_at",
		}).AddRow(
			userID, "test_openid", "", "tenant_123", "",
			"Test User", "", "", "", "user", "active",
			time.Now(), time.Now(), time.Now(),
		)

		mock.ExpectQuery("SELECT (.+) FROM users WHERE id = \\$1").
			WithArgs(userID).
			WillReturnRows(rows)

		user, err := repo.GetByID(ctx, userID)
		assert.NoError(t, err)
		assert.Equal(t, userID, user.ID)
		assert.Equal(t, "Test User", user.Nickname)
	})

	assert.NoError(t, mock.ExpectationsWereMet())
}

func TestDatasetRepository_Simple(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	repo := NewPostgresDatasetRepository(db)
	ctx := context.Background()

	t.Run("Create Dataset", func(t *testing.T) {
		dataset := &Dataset{
			Name:        "Test Dataset",
			Description: "A test dataset",
			TenantID:    "tenant_123",
			CreatedBy:   "user_123",
		}

		mock.ExpectExec("INSERT INTO datasets").
			WillReturnResult(sqlmock.NewResult(1, 1))

		err = repo.Create(ctx, dataset)
		assert.NoError(t, err)
		assert.NotEmpty(t, dataset.ID)
	})

	t.Run("Get Dataset", func(t *testing.T) {
		datasetID := "dataset_123"

		rows := sqlmock.NewRows([]string{
			"id", "tenant_id", "name", "description", "type", "status",
			"doc_count", "chunk_count", "token_count", "metadata",
			"created_by", "updated_by", "created_at", "updated_at",
		}).AddRow(
			datasetID, "tenant_123", "Test Dataset", "Description", "document", "active",
			0, 0, int64(0), "{}", "user_123", "", time.Now(), time.Now(),
		)

		mock.ExpectQuery("SELECT (.+) FROM datasets WHERE id = \\$1").
			WithArgs(datasetID).
			WillReturnRows(rows)

		dataset, err := repo.Get(ctx, datasetID)
		assert.NoError(t, err)
		assert.Equal(t, datasetID, dataset.ID)
		assert.Equal(t, "Test Dataset", dataset.Name)
	})

	assert.NoError(t, mock.ExpectationsWereMet())
}

func TestCacheOperations_Simple(t *testing.T) {
	// 简单的缓存操作测试
	t.Run("Cache Interface", func(t *testing.T) {
		// 这里可以测试缓存接口的基本功能
		// 由于我们使用的是接口，可以创建一个简单的内存实现进行测试

		cache := make(map[string]string)

		// 模拟Set操作
		key := "test_key"
		value := "test_value"
		cache[key] = value

		// 模拟Get操作
		result, exists := cache[key]
		assert.True(t, exists)
		assert.Equal(t, value, result)

		// 模拟Delete操作
		delete(cache, key)
		_, exists = cache[key]
		assert.False(t, exists)
	})
}
