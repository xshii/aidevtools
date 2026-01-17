"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const pluginManager_1 = require("../src/pluginManager");
describe('PluginManager', () => {
    let manager;
    beforeEach(() => {
        manager = new pluginManager_1.PluginManager();
    });
    describe('register', () => {
        it('should register a renderer', () => {
            const renderer = {
                type: 'chart',
                render: () => '<div>chart</div>',
            };
            manager.register(renderer);
            expect(manager.hasRenderer('chart')).toBe(true);
        });
        it('should overwrite existing renderer', () => {
            const renderer1 = {
                type: 'chart',
                render: () => '<div>old</div>',
            };
            const renderer2 = {
                type: 'chart',
                render: () => '<div>new</div>',
            };
            manager.register(renderer1);
            manager.register(renderer2);
            const artifact = { type: 'chart', data: {} };
            expect(manager.render(artifact)).toBe('<div>new</div>');
        });
    });
    describe('unregister', () => {
        it('should unregister a renderer', () => {
            const renderer = {
                type: 'table',
                render: () => '<table></table>',
            };
            manager.register(renderer);
            manager.unregister('table');
            expect(manager.hasRenderer('table')).toBe(false);
        });
    });
    describe('getRenderer', () => {
        it('should return renderer if exists', () => {
            const renderer = {
                type: 'json',
                render: () => '{}',
            };
            manager.register(renderer);
            expect(manager.getRenderer('json')).toBe(renderer);
        });
        it('should return undefined if not exists', () => {
            expect(manager.getRenderer('unknown')).toBeUndefined();
        });
    });
    describe('render', () => {
        it('should render artifact using registered renderer', () => {
            const renderer = {
                type: 'markdown',
                render: (artifact) => `<p>${artifact.data.content}</p>`,
            };
            manager.register(renderer);
            const artifact = {
                type: 'markdown',
                data: { content: 'hello' },
            };
            expect(manager.render(artifact)).toBe('<p>hello</p>');
        });
        it('should use fallback for unknown type', () => {
            const artifact = {
                type: 'unknown',
                data: { foo: 'bar' },
            };
            const result = manager.render(artifact);
            expect(result).toContain('Unknown artifact type');
            expect(result).toContain('foo');
        });
    });
    describe('listTypes', () => {
        it('should list all registered types', () => {
            manager.register({ type: 'chart', render: () => '' });
            manager.register({ type: 'table', render: () => '' });
            manager.register({ type: 'json', render: () => '' });
            const types = manager.listTypes();
            expect(types).toContain('chart');
            expect(types).toContain('table');
            expect(types).toContain('json');
            expect(types.length).toBe(3);
        });
        it('should return empty array when no renderers', () => {
            expect(manager.listTypes()).toEqual([]);
        });
    });
});
//# sourceMappingURL=pluginManager.test.js.map