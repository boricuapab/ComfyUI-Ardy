import { app } from '../../../scripts/app.js';
import { ArdyViewer } from './live_preview.js';
import { CameraPreview, pickFile } from './camera_preview.js';

const style = document.createElement('style');
style.textContent = `
    .pointer-events-none.absolute.size-full > .pointer-events-auto.absolute.top-0.left-0:has(.pi-play, .pi-pause):has([role="slider"]) {
        top: auto !important; bottom: 8px; padding: 8px 0;
        background: rgba(23, 29, 37, .85); border-radius: 6px;
    }
`;
document.head.append(style);

app.registerExtension({
    name: 'ComfyUI.Ardy',
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === 'ArdyLoadCamera') {
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                original?.apply(this, arguments);
                this.addWidget('button', 'Load camera JSON…', null, async () => {
                    const file = await pickFile('.json');
                    if (!file) return;
                    const value = await file.text();
                    try {
                        if (!Array.isArray(JSON.parse(value))) throw new Error('Expected a camera track list.');
                        this.widgets.find(w => w.name === 'camera_track').value = value;
                        this.graph?.change();
                    } catch (error) { alert(error.message); }
                });
            };
            return;
        }
        if (nodeData.name === 'ArdyCameraRecorder') {
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                original?.apply(this, arguments);
                const track = this.widgets.find(w => w.name === 'camera_track');
                track.hidden = true; track.options.hidden = true;
                track.computeSize = () => [0, -4]; track.draw = () => {}; track.mouse = () => true;
                const recording = this.widgets.find(w => w.name === 'recording_id');
                recording.hidden = true; recording.options.hidden = true;
                recording.computeSize = () => [0, -4]; recording.draw = () => {}; recording.mouse = () => true;
                const viewer = this.ardyCamera = new CameraPreview(this);
                const widget = this.addDOMWidget('ardy_camera', 'custom', viewer.root, {
                    serialize: false, hideOnZoom: false, getMinHeight: () => 540, getHeight: () => 540,
                });
                widget.computeSize = () => [650, 540]; this.setSize([700, 900]);
                const executed = this.onExecuted;
                this.onExecuted = function (message) {
                    executed?.apply(this, arguments);
                    const file = message.ardy_camera?.[0];
                    if (message.ardy_recording_id?.[0]) {
                        recording.value = message.ardy_recording_id[0];
                        this.graph?.change();
                    }
                    if (file) {
                        this.properties.ardy_camera_file = file;
                        viewer.load(file).catch(error => { viewer.status.textContent = error.message; });
                    }
                };
                const configured = this.onConfigure;
                this.onConfigure = function () {
                    configured?.apply(this, arguments);
                    const file = this.properties.ardy_camera_file;
                    if (file) viewer.load(file).catch(error => { viewer.status.textContent = error.message; });
                };
                const removed = this.onRemoved;
                this.onRemoved = function () { viewer.dispose(); return removed?.apply(this, arguments); };
            };
            return;
        }
        if (nodeData.name !== 'ArdyLivePreview') return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const control = this.widgets.find(w => w.name === 'control_id');
            control.value = crypto.randomUUID();
            control.hidden = true; control.options.hidden = true;
            control.computeSize = () => [0, -4];
            control.draw = () => {}; control.mouse = () => true;
            this.ardyViewer = new ArdyViewer(this);
            const widget = this.addDOMWidget('ardy_preview', 'custom', this.ardyViewer.root, {
                serialize: false, hideOnZoom: false, getMinHeight: () => 540, getHeight: () => 540,
            });
            widget.computeSize = () => [650, 540];
            this.setSize([700, 900]);
            const onRemoved = this.onRemoved;
            this.onRemoved = function () { this.ardyViewer.dispose(); return onRemoved?.apply(this, arguments); };
            const executed = this.onExecuted;
            this.onExecuted = function (message) {
                executed?.apply(this, arguments);
                if (message.ardy_status) this.ardyViewer.status.textContent = message.ardy_status[0];
            };
        };
    },
});
