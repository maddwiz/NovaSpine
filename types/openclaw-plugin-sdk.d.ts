declare module "openclaw/plugin-sdk/memory-core" {
  export type OpenClawLogger = {
    info(message: string): void;
    warn(message: string): void;
    error?(message: string): void;
  };

  export type OpenClawPluginApi = {
    pluginConfig: unknown;
    config: unknown;
    logger: OpenClawLogger;
    resolvePath(target: string): string;
    registerService(service: {
      id: string;
      start?(args: { logger: OpenClawLogger }): void | Promise<void>;
      stop?(): void | Promise<void>;
    }): void;
    registerTool(tool: unknown, options?: unknown): void;
    registerCli(handler: (args: { program: any }) => void, options?: unknown): void;
    registerContextEngine(name: string, factory: () => unknown): void;
    registerCommand(command: unknown): void;
    on(event: string, handler: (payload: any) => unknown): void;
  };
}
