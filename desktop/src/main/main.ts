import { app, BrowserWindow, Menu, ipcMain, shell, dialog, Tray, nativeImage } from 'electron';
import { autoUpdater } from 'electron-updater';
import windowStateKeeper from 'electron-window-state';
import Store from 'electron-store';
import * as path from 'path';
import * as notifier from 'node-notifier';

// Initialize store for persistent settings
const store = new Store();

class ChatbotApp {
  private mainWindow: BrowserWindow | null = null;
  private tray: Tray | null = null;
  private isQuitting = false;

  constructor() {
    this.setupApp();
  }

  private setupApp(): void {
    // Handle app events
    app.whenReady().then(() => {
      this.createMainWindow();
      this.setupMenu();
      this.setupTray();
      this.setupAutoUpdater();
      this.setupIpcHandlers();

      app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
          this.createMainWindow();
        }
      });
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        app.quit();
      }
    });

    app.on('before-quit', () => {
      this.isQuitting = true;
    });

    // Security: Prevent new window creation
    app.on('web-contents-created', (event, contents) => {
      contents.on('new-window', (navigationEvent, navigationURL) => {
        navigationEvent.preventDefault();
        shell.openExternal(navigationURL);
      });
    });
  }

  private createMainWindow(): void {
    // Manage window state
    const mainWindowState = windowStateKeeper({
      defaultWidth: 1200,
      defaultHeight: 800,
    });

    // Create the browser window
    this.mainWindow = new BrowserWindow({
      x: mainWindowState.x,
      y: mainWindowState.y,
      width: mainWindowState.width,
      height: mainWindowState.height,
      minWidth: 800,
      minHeight: 600,
      show: false,
      icon: this.getAppIcon(),
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: true,
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    });

    // Let windowStateKeeper manage the window
    mainWindowState.manage(this.mainWindow);

    // Load the app
    if (process.env.NODE_ENV === 'development') {
      this.mainWindow.loadURL('http://localhost:3000');
      this.mainWindow.webContents.openDevTools();
    } else {
      this.mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    }

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow?.show();
      
      // Focus on window
      if (process.env.NODE_ENV === 'development') {
        this.mainWindow?.focus();
      }
    });

    // Handle window close
    this.mainWindow.on('close', (event) => {
      if (!this.isQuitting && process.platform === 'darwin') {
        event.preventDefault();
        this.mainWindow?.hide();
      }
    });

    // Handle window closed
    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });

    // Handle external links
    this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: 'deny' };
    });
  }

  private setupMenu(): void {
    const template: Electron.MenuItemConstructorOptions[] = [
      {
        label: 'File',
        submenu: [
          {
            label: 'New Chat',
            accelerator: 'CmdOrCtrl+N',
            click: () => {
              this.mainWindow?.webContents.send('new-chat');
            },
          },
          {
            label: 'Export Chat',
            accelerator: 'CmdOrCtrl+E',
            click: () => {
              this.exportChat();
            },
          },
          { type: 'separator' },
          {
            label: 'Preferences',
            accelerator: 'CmdOrCtrl+,',
            click: () => {
              this.mainWindow?.webContents.send('open-preferences');
            },
          },
          { type: 'separator' },
          {
            role: 'quit',
          },
        ],
      },
      {
        label: 'Edit',
        submenu: [
          { role: 'undo' },
          { role: 'redo' },
          { type: 'separator' },
          { role: 'cut' },
          { role: 'copy' },
          { role: 'paste' },
          { role: 'selectall' },
        ],
      },
      {
        label: 'View',
        submenu: [
          { role: 'reload' },
          { role: 'forceReload' },
          { role: 'toggleDevTools' },
          { type: 'separator' },
          { role: 'resetZoom' },
          { role: 'zoomIn' },
          { role: 'zoomOut' },
          { type: 'separator' },
          { role: 'togglefullscreen' },
        ],
      },
      {
        label: 'Window',
        submenu: [
          { role: 'minimize' },
          { role: 'close' },
          ...(process.platform === 'darwin'
            ? [
                { type: 'separator' as const },
                { role: 'front' as const },
              ]
            : []),
        ],
      },
      {
        role: 'help',
        submenu: [
          {
            label: 'About AI Chatbot',
            click: () => {
              this.showAboutDialog();
            },
          },
          {
            label: 'Learn More',
            click: () => {
              shell.openExternal('https://github.com/your-org/chatbot');
            },
          },
        ],
      },
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  }

  private setupTray(): void {
    const trayIcon = this.getAppIcon();
    this.tray = new Tray(trayIcon);

    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Show AI Chatbot',
        click: () => {
          this.mainWindow?.show();
        },
      },
      {
        label: 'New Chat',
        click: () => {
          this.mainWindow?.webContents.send('new-chat');
          this.mainWindow?.show();
        },
      },
      { type: 'separator' },
      {
        label: 'Quit',
        click: () => {
          this.isQuitting = true;
          app.quit();
        },
      },
    ]);

    this.tray.setToolTip('AI Chatbot');
    this.tray.setContextMenu(contextMenu);

    // Handle tray click
    this.tray.on('click', () => {
      if (this.mainWindow?.isVisible()) {
        this.mainWindow.hide();
      } else {
        this.mainWindow?.show();
      }
    });
  }

  private setupAutoUpdater(): void {
    if (process.env.NODE_ENV === 'production') {
      autoUpdater.checkForUpdatesAndNotify();

      autoUpdater.on('update-available', () => {
        notifier.notify({
          title: 'AI Chatbot',
          message: 'A new update is available. It will be downloaded in the background.',
        });
      });

      autoUpdater.on('update-downloaded', () => {
        notifier.notify({
          title: 'AI Chatbot',
          message: 'Update downloaded. The application will restart to apply the update.',
        });

        // Restart app to apply update
        setTimeout(() => {
          autoUpdater.quitAndInstall();
        }, 5000);
      });
    }
  }

  private setupIpcHandlers(): void {
    // Handle app info requests
    ipcMain.handle('get-app-info', () => {
      return {
        name: app.getName(),
        version: app.getVersion(),
        platform: process.platform,
        arch: process.arch,
      };
    });

    // Handle settings
    ipcMain.handle('get-setting', (event, key: string) => {
      return store.get(key);
    });

    ipcMain.handle('set-setting', (event, key: string, value: any) => {
      store.set(key, value);
    });

    // Handle notifications
    ipcMain.handle('show-notification', (event, options: any) => {
      notifier.notify({
        title: options.title || 'AI Chatbot',
        message: options.message,
        icon: this.getAppIcon(),
        sound: options.sound !== false,
      });
    });

    // Handle file operations
    ipcMain.handle('show-save-dialog', async (event, options: any) => {
      const result = await dialog.showSaveDialog(this.mainWindow!, options);
      return result;
    });

    ipcMain.handle('show-open-dialog', async (event, options: any) => {
      const result = await dialog.showOpenDialog(this.mainWindow!, options);
      return result;
    });

    // Handle window operations
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

    // Handle external links
    ipcMain.handle('open-external', (event, url: string) => {
      shell.openExternal(url);
    });
  }

  private getAppIcon(): nativeImage {
    const iconPath = path.join(__dirname, '../../assets/icon.png');
    return nativeImage.createFromPath(iconPath);
  }

  private async exportChat(): Promise<void> {
    try {
      const result = await dialog.showSaveDialog(this.mainWindow!, {
        title: 'Export Chat',
        defaultPath: `chat-export-${new Date().toISOString().split('T')[0]}.json`,
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'Text Files', extensions: ['txt'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });

      if (!result.canceled && result.filePath) {
        this.mainWindow?.webContents.send('export-chat', result.filePath);
      }
    } catch (error) {
      console.error('Export chat error:', error);
    }
  }

  private showAboutDialog(): void {
    dialog.showMessageBox(this.mainWindow!, {
      type: 'info',
      title: 'About AI Chatbot',
      message: 'AI Chatbot Desktop',
      detail: `Version: ${app.getVersion()}\nElectron: ${process.versions.electron}\nNode.js: ${process.versions.node}\nChromium: ${process.versions.chrome}`,
      buttons: ['OK'],
    });
  }
}

// Create app instance
new ChatbotApp();
