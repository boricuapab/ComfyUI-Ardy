import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

export async function queuePreview(node) {
    const graph = await app.graphToPrompt();
    const output = {};
    const visit = id => {
        if (output[id]) return;
        const entry = graph.output[id];
        if (!entry) throw new Error('Preview node is not executable.');
        if (entry.class_type === 'ArdyCameraRecorder' && entry.inputs.recording_id) delete entry.inputs.motion;
        output[id] = entry;
        for (const input of Object.values(entry.inputs)) {
            if (Array.isArray(input) && input.length === 2 && graph.output[input[0]]) visit(String(input[0]));
        }
    };
    visit(String(node.id));
    await api.queuePrompt(0, {output, workflow: graph.workflow});
}
