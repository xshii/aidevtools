import { PluginManager } from '../src/pluginManager';
import { ArtifactRenderer, Artifact } from '../src/types';

describe('PluginManager', () => {
  let manager: PluginManager;

  beforeEach(() => {
    manager = new PluginManager();
  });

  describe('register', () => {
    it('should register a renderer', () => {
      const renderer: ArtifactRenderer = {
        type: 'chart',
        render: () => '<div>chart</div>',
      };

      manager.register(renderer);

      expect(manager.hasRenderer('chart')).toBe(true);
    });

    it('should overwrite existing renderer', () => {
      const renderer1: ArtifactRenderer = {
        type: 'chart',
        render: () => '<div>old</div>',
      };
      const renderer2: ArtifactRenderer = {
        type: 'chart',
        render: () => '<div>new</div>',
      };

      manager.register(renderer1);
      manager.register(renderer2);

      const artifact: Artifact = { type: 'chart', data: {} };
      expect(manager.render(artifact)).toBe('<div>new</div>');
    });
  });

  describe('unregister', () => {
    it('should unregister a renderer', () => {
      const renderer: ArtifactRenderer = {
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
      const renderer: ArtifactRenderer = {
        type: 'json',
        render: () => '{}',
      };

      manager.register(renderer);

      expect(manager.getRenderer('json')).toBe(renderer);
    });

    it('should return undefined if not exists', () => {
      expect(manager.getRenderer('unknown' as any)).toBeUndefined();
    });
  });

  describe('render', () => {
    it('should render artifact using registered renderer', () => {
      const renderer: ArtifactRenderer = {
        type: 'markdown',
        render: (artifact) => `<p>${(artifact.data as any).content}</p>`,
      };

      manager.register(renderer);

      const artifact: Artifact = {
        type: 'markdown',
        data: { content: 'hello' },
      };

      expect(manager.render(artifact)).toBe('<p>hello</p>');
    });

    it('should use fallback for unknown type', () => {
      const artifact: Artifact = {
        type: 'unknown' as any,
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
