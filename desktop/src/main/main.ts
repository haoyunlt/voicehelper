import { 
  app, 
  BrowserWindow, 
  ipcMain, 
  Menu, 
  Tray, 
  globalShortcut, 
  dialog, 
  shell,
  nativeTheme,
  protocol,
  session
} from 'electron';
import { autoUpdater } from 'electron-updater';
import * as path from 'path';
import * as fs from 'fs';
import { VoiceHelperSDK } from '@voicehelper/sdk';

// 应用配置
interface AppConfig {
  apiKey: string;
  theme: 'light' | 'dark' | 'system';
  autoStart: boolean;
  minimizeToTray: boolean;
  globalShortcuts: {
    toggleWindow: string;
    startVoice: string;
    stopVoice: string;
  };
  windowBounds: {
    width: number;
    height: number;
    x?: number;
    y?: number;
  };
}

class VoiceHelperApp {
  private mainWindow: BrowserWindow | null = null;
  private tray: Tray | null = null;
  private voiceSDK: VoiceHelperSDK | null = null;
  private config: AppConfig;
  private configPath: string;

  constructor() {
    this.configPath = path.join(app.getPath('userData'), 'config.json');
    this.config = this.loadConfig();
    this.setupApp();
  }

  private loadConfig(): AppConfig {
    const defaultConfig: AppConfig = {
      apiKey: '',
      theme: 'system',
      autoStart: false,
      minimizeToTray: true,
      globalShortcuts: {
        toggleWindow: 'CommandOrControl+Shift+V',
        startVoice: 'CommandOrControl+Shift+S',
        stopVoice: 'CommandOrControl+Shift+E'
      },
      windowBounds: {
        width: 1200,
        height: 800
      }
    };

    try {
      if (fs.existsSync(this.configPath)) {
        const savedConfig = JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
        return { ...defaultConfig, ...savedConfig };
      }
    } catch (error) {
      console.error('Failed to load config:', error);
    }

    return defaultConfig;
  }

  private saveConfig(): void {
    try {
      fs.writeFileSync(this.configPath, JSON.stringify(this.config, null, 2));
    } catch (error) {
      console.error('Failed to save config:', error);
    }
  }

  private setupApp(): void {
    // 设置应用协议
    protocol.registerSchemesAsPrivileged([
      { scheme: 'voicehelper', privileges: { secure: true, standard: true } }
    ]);

    // 应用事件监听
    app.whenReady().then(() => {
      this.createWindow();
      this.setupTray();
      this.setupGlobalShortcuts();
      this.setupAutoUpdater();
      this.setupIPC();
      this.initializeSDK();
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        app.quit();
      }
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        this.createWindow();
      }
    });

    app.on('before-quit', () => {
      this.saveConfig();
    });

    // 设置主题
    nativeTheme.themeSource = this.config.theme;
  }

  private createWindow(): void {
    // 创建主窗口
    this.mainWindow = new BrowserWindow({
      width: this.config.windowBounds.width,
      height: this.config.windowBounds.height,
      x: this.config.windowBounds.x,
      y: this.config.windowBounds.y,
      minWidth: 800,
      minHeight: 600,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, '../preload/preload.js'),
        webSecurity: true
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
      show: false,
      icon: path.join(__dirname, '../../assets/icon.png')
    });

    // 加载应用
    if (process.env.NODE_ENV === 'development') {
      this.mainWindow.loadURL('http://localhost:3000');
      this.mainWindow.webContents.openDevTools();
    } else {
      this.mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    }

    // 窗口事件
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow?.show();
      
      if (process.env.NODE_ENV === 'development') {
        this.mainWindow?.webContents.openDevTools();
      }
    });

    this.mainWindow.on('close', (event) => {
      if (this.config.minimizeToTray && this.tray) {
        event.preventDefault();
        this.mainWindow?.hide();
      } else {
        // 保存窗口位置
        const bounds = this.mainWindow?.getBounds();
        if (bounds) {
          this.config.windowBounds = bounds;
          this.saveConfig();
        }
      }
    });

    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });

    // 外部链接在默认浏览器中打开
    this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: 'deny' };
    });

    // 设置菜单
    this.setupMenu();
  }

  private setupTray(): void {
    const iconPath = path.join(__dirname, '../../assets/tray-icon.png');
    this.tray = new Tray(iconPath);

    const contextMenu = Menu.buildFromTemplate([
      {
        label: '显示窗口',
        click: () => {
          this.mainWindow?.show();
        }
      },
      {
        label: '开始语音',
        click: () => {
          this.mainWindow?.webContents.send('start-voice-recording');
        }
      },
      {
        label: '停止语音',
        click: () => {
          this.mainWindow?.webContents.send('stop-voice-recording');
        }
      },
      { type: 'separator' },
      {
        label: '设置',
        click: () => {
          this.mainWindow?.show();
          this.mainWindow?.webContents.send('navigate-to-settings');
        }
      },
      {
        label: '关于',
        click: () => {
          dialog.showMessageBox(this.mainWindow!, {
            type: 'info',
            title: '关于 VoiceHelper',
            message: 'VoiceHelper Desktop',
            detail: `版本: ${app.getVersion()}\n企业级智能对话和语音处理平台`
          });
        }
      },
      { type: 'separator' },
      {
        label: '退出',
        click: () => {
          app.quit();
        }
      }
    ]);

    this.tray.setContextMenu(contextMenu);
    this.tray.setToolTip('VoiceHelper - 智能语音助手');

    this.tray.on('click', () => {
      if (this.mainWindow?.isVisible()) {
        this.mainWindow.hide();
      } else {
        this.mainWindow?.show();
      }
    });
  }

  private setupGlobalShortcuts(): void {
    // 注册全局快捷键
    globalShortcut.register(this.config.globalShortcuts.toggleWindow, () => {
      if (this.mainWindow?.isVisible()) {
        this.mainWindow.hide();
      } else {
        this.mainWindow?.show();
        this.mainWindow?.focus();
      }
    });

    globalShortcut.register(this.config.globalShortcuts.startVoice, () => {
      this.mainWindow?.webContents.send('global-shortcut-start-voice');
    });

    globalShortcut.register(this.config.globalShortcuts.stopVoice, () => {
      this.mainWindow?.webContents.send('global-shortcut-stop-voice');
    });
  }

  private setupMenu(): void {
    const template: Electron.MenuItemConstructorOptions[] = [
      {
        label: '文件',
        submenu: [
          {
            label: '新建对话',
            accelerator: 'CmdOrCtrl+N',
            click: () => {
              this.mainWindow?.webContents.send('new-conversation');
            }
          },
          {
            label: '导入对话',
            click: async () => {
              const result = await dialog.showOpenDialog(this.mainWindow!, {
                properties: ['openFile'],
                filters: [
                  { name: 'JSON Files', extensions: ['json'] }
                ]
              });

              if (!result.canceled && result.filePaths.length > 0) {
                this.mainWindow?.webContents.send('import-conversation', result.filePaths[0]);
              }
            }
          },
          {
            label: '导出对话',
            accelerator: 'CmdOrCtrl+E',
            click: async () => {
              const result = await dialog.showSaveDialog(this.mainWindow!, {
                filters: [
                  { name: 'JSON Files', extensions: ['json'] }
                ]
              });

              if (!result.canceled && result.filePath) {
                this.mainWindow?.webContents.send('export-conversation', result.filePath);
              }
            }
          },
          { type: 'separator' },
          {
            label: '退出',
            accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
            click: () => {
              app.quit();
            }
          }
        ]
      },
      {
        label: '编辑',
        submenu: [
          { role: 'undo', label: '撤销' },
          { role: 'redo', label: '重做' },
          { type: 'separator' },
          { role: 'cut', label: '剪切' },
          { role: 'copy', label: '复制' },
          { role: 'paste', label: '粘贴' },
          { role: 'selectall', label: '全选' }
        ]
      },
      {
        label: '视图',
        submenu: [
          { role: 'reload', label: '重新加载' },
          { role: 'forceReload', label: '强制重新加载' },
          { role: 'toggleDevTools', label: '开发者工具' },
          { type: 'separator' },
          { role: 'resetZoom', label: '实际大小' },
          { role: 'zoomIn', label: '放大' },
          { role: 'zoomOut', label: '缩小' },
          { type: 'separator' },
          { role: 'togglefullscreen', label: '全屏' }
        ]
      },
      {
        label: '语音',
        submenu: [
          {
            label: '开始录音',
            accelerator: this.config.globalShortcuts.startVoice,
            click: () => {
              this.mainWindow?.webContents.send('start-voice-recording');
            }
          },
          {
            label: '停止录音',
            accelerator: this.config.globalShortcuts.stopVoice,
            click: () => {
              this.mainWindow?.webContents.send('stop-voice-recording');
            }
          },
          { type: 'separator' },
          {
            label: '语音设置',
            click: () => {
              this.mainWindow?.webContents.send('open-voice-settings');
            }
          }
        ]
      },
      {
        label: '窗口',
        submenu: [
          { role: 'minimize', label: '最小化' },
          { role: 'close', label: '关闭' },
          {
            label: '置顶',
            type: 'checkbox',
            click: (menuItem) => {
              this.mainWindow?.setAlwaysOnTop(menuItem.checked);
            }
          }
        ]
      },
      {
        label: '帮助',
        submenu: [
          {
            label: '开发者文档',
            click: () => {
              shell.openExternal('https://docs.voicehelper.ai');
            }
          },
          {
            label: '快捷键',
            click: () => {
              this.showShortcutsDialog();
            }
          },
          {
            label: '检查更新',
            click: () => {
              autoUpdater.checkForUpdatesAndNotify();
            }
          },
          { type: 'separator' },
          {
            label: '关于',
            click: () => {
              dialog.showMessageBox(this.mainWindow!, {
                type: 'info',
                title: '关于 VoiceHelper',
                message: 'VoiceHelper Desktop',
                detail: `版本: ${app.getVersion()}\n企业级智能对话和语音处理平台\n\n© 2025 VoiceHelper AI. All rights reserved.`
              });
            }
          }
        ]
      }
    ];

    // macOS 特殊处理
    if (process.platform === 'darwin') {
      template.unshift({
        label: app.getName(),
        submenu: [
          { role: 'about', label: '关于 VoiceHelper' },
          { type: 'separator' },
          { role: 'services', label: '服务' },
          { type: 'separator' },
          { role: 'hide', label: '隐藏 VoiceHelper' },
          { role: 'hideOthers', label: '隐藏其他' },
          { role: 'unhide', label: '显示全部' },
          { type: 'separator' },
          { role: 'quit', label: '退出 VoiceHelper' }
        ]
      });
    }

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  }

  private setupAutoUpdater(): void {
    autoUpdater.checkForUpdatesAndNotify();

    autoUpdater.on('update-available', () => {
      dialog.showMessageBox(this.mainWindow!, {
        type: 'info',
        title: '更新可用',
        message: '发现新版本，正在下载...',
        buttons: ['确定']
      });
    });

    autoUpdater.on('update-downloaded', () => {
      dialog.showMessageBox(this.mainWindow!, {
        type: 'info',
        title: '更新就绪',
        message: '更新已下载完成，重启应用以应用更新。',
        buttons: ['重启', '稍后']
      }).then((result) => {
        if (result.response === 0) {
          autoUpdater.quitAndInstall();
        }
      });
    });
  }

  private setupIPC(): void {
    // 配置相关
    ipcMain.handle('get-config', () => {
      return this.config;
    });

    ipcMain.handle('update-config', (_, newConfig: Partial<AppConfig>) => {
      this.config = { ...this.config, ...newConfig };
      this.saveConfig();
      
      // 应用主题更改
      if (newConfig.theme) {
        nativeTheme.themeSource = newConfig.theme;
      }
      
      return this.config;
    });

    // 窗口控制
    ipcMain.handle('minimize-window', () => {
      this.mainWindow?.minimize();
    });

    ipcMain.handle('maximize-window', () => {
      if (this.mainWindow?.isMaximized()) {
        this.mainWindow.unmaximize();
      } else {
        this.mainWindow?.maximize();
      }
    });

    ipcMain.handle('close-window', () => {
      this.mainWindow?.close();
    });

    // 文件操作
    ipcMain.handle('show-save-dialog', async (_, options) => {
      const result = await dialog.showSaveDialog(this.mainWindow!, options);
      return result;
    });

    ipcMain.handle('show-open-dialog', async (_, options) => {
      const result = await dialog.showOpenDialog(this.mainWindow!, options);
      return result;
    });

    ipcMain.handle('write-file', async (_, filePath: string, data: string) => {
      try {
        fs.writeFileSync(filePath, data, 'utf8');
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    });

    ipcMain.handle('read-file', async (_, filePath: string) => {
      try {
        const data = fs.readFileSync(filePath, 'utf8');
        return { success: true, data };
      } catch (error) {
        return { success: false, error: error.message };
      }
    });

    // 系统信息
    ipcMain.handle('get-app-version', () => {
      return app.getVersion();
    });

    ipcMain.handle('get-platform', () => {
      return process.platform;
    });

    // 外部链接
    ipcMain.handle('open-external', (_, url: string) => {
      shell.openExternal(url);
    });

    // 通知
    ipcMain.handle('show-notification', (_, options) => {
      const { Notification } = require('electron');
      if (Notification.isSupported()) {
        new Notification(options).show();
      }
    });

    // 语音相关
    ipcMain.handle('start-voice-recording', () => {
      // 这里可以集成系统级语音录制
      this.mainWindow?.webContents.send('voice-recording-started');
    });

    ipcMain.handle('stop-voice-recording', () => {
      this.mainWindow?.webContents.send('voice-recording-stopped');
    });

    // API 调用
    ipcMain.handle('api-call', async (_, method: string, endpoint: string, data?: any) => {
      if (!this.voiceSDK) {
        return { success: false, error: 'SDK not initialized' };
      }

      try {
        // 这里根据不同的方法调用相应的SDK方法
        let result;
        switch (method) {
          case 'chat':
            result = await this.voiceSDK.createChatCompletion(data);
            break;
          case 'transcribe':
            result = await this.voiceSDK.transcribeAudio(data);
            break;
          case 'synthesize':
            result = await this.voiceSDK.synthesizeText(data);
            break;
          default:
            throw new Error(`Unknown method: ${method}`);
        }
        
        return { success: true, data: result };
      } catch (error) {
        return { success: false, error: error.message };
      }
    });
  }

  private initializeSDK(): void {
    if (this.config.apiKey) {
      this.voiceSDK = new VoiceHelperSDK({
        apiKey: this.config.apiKey,
        debug: process.env.NODE_ENV === 'development'
      });
    }
  }

  private showShortcutsDialog(): void {
    const shortcuts = [
      { action: '切换窗口显示/隐藏', shortcut: this.config.globalShortcuts.toggleWindow },
      { action: '开始语音录制', shortcut: this.config.globalShortcuts.startVoice },
      { action: '停止语音录制', shortcut: this.config.globalShortcuts.stopVoice },
      { action: '新建对话', shortcut: 'Ctrl/Cmd+N' },
      { action: '导出对话', shortcut: 'Ctrl/Cmd+E' },
      { action: '退出应用', shortcut: 'Ctrl/Cmd+Q' }
    ];

    const message = shortcuts
      .map(item => `${item.action}: ${item.shortcut}`)
      .join('\n');

    dialog.showMessageBox(this.mainWindow!, {
      type: 'info',
      title: '快捷键',
      message: '全局快捷键',
      detail: message
    });
  }

  public updateSDK(apiKey: string): void {
    this.config.apiKey = apiKey;
    this.saveConfig();
    this.initializeSDK();
  }
}

// 创建应用实例
const voiceHelperApp = new VoiceHelperApp();

// 导出用于测试
export default voiceHelperApp;