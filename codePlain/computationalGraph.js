
const context =
	await navigator.ml.createContext({powerPreference: 'low-power'});

// Use tensors in 4 dimensions.
const TENSOR_DIMS = [1, 2, 2, 2];
const TENSOR_SIZE = 8;

const builder = new MLGraphBuilder(context);

// Create MLOperandDescriptor object.
const desc = {type: 'float32', dimensions: TENSOR_DIMS};

// constant1 is a constant MLOperand with the value 0.5.
const constantBuffer1 = new Float32Array(TENSOR_SIZE).fill(1);
const input1 = builder.constant(desc, constantBuffer1);

// input1 is one of the input MLOperands.
// Its value will be set before execution.
const input2 = builder.input('input1', desc);

// constant2 is another constant MLOperand with the value 0.5.
const constantBuffer3 = new Float32Array(TENSOR_SIZE).fill(3);
const input3 = builder.constant(desc, constantBuffer3);

// input2 is another input MLOperand. Its value will be set before execution.
const input4 = builder.input('input2', desc);

// intermediateOutput1 is the output of the first Add operation.
const intermediateOutput1 = builder.add(input1, input2);

// intermediateOutput2 is the output of the second Add operation.
const intermediateOutput2 = builder.add(input3, input4);

// output is the output MLOperand of the Mul operation.
const output = builder.mul(intermediateOutput1, intermediateOutput2);

// Compile the constructed graph.
const graph = await builder.build({'output': output});

// Setup the input buffers with value 1.
const inputBuffer3 = new Float32Array(TENSOR_SIZE).fill(2);
const inputBuffer4 = new Float32Array(TENSOR_SIZE).fill(4);
const outputBuffer = new Float32Array(TENSOR_SIZE);

// Execute the compiled graph with the specified inputs.
const inputs = {
	'input1': inputBuffer3,
	'input2': inputBuffer4,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

document.body.append(
	document.createElement('div').appendChild(document.createTextNode(results.outputs.output))
)
