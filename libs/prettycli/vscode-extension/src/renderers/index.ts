export { getBaseHtml, escapeHtml } from './base';
export { ChartRenderer } from './chart';
export { TableRenderer } from './table';
export { FileRenderer } from './file';
export { DiffRenderer } from './diff';
export { WebRenderer } from './web';
export { MarkdownRenderer } from './markdown';
export { JsonRenderer } from './json';
export { ImageRenderer } from './image';

import { pluginManager } from '../pluginManager';
import { ChartRenderer } from './chart';
import { TableRenderer } from './table';
import { FileRenderer } from './file';
import { DiffRenderer } from './diff';
import { WebRenderer } from './web';
import { MarkdownRenderer } from './markdown';
import { JsonRenderer } from './json';
import { ImageRenderer } from './image';

/**
 * Register all built-in renderers
 */
export function registerBuiltinRenderers(): void {
    pluginManager.register(new ChartRenderer());
    pluginManager.register(new TableRenderer());
    pluginManager.register(new FileRenderer());
    pluginManager.register(new DiffRenderer());
    pluginManager.register(new WebRenderer());
    pluginManager.register(new MarkdownRenderer());
    pluginManager.register(new JsonRenderer());
    pluginManager.register(new ImageRenderer());
}
