function getMLcontext(){
	return navigator.ml.createContext();
}

export function multiplyMatrices(dim1,dim2,dimRes,initial){
	const context=getMLcontext()
	const builder = new MLGraphBuilder(context);
	const descA = {type: 'float32', dimensions: dim1};
	const a = builder.input('a', descA);
	const descB = {type: 'float32', dimensions: dim2};
	const bufferB = new Float32Array(sizeOfShape(descB.dimensions)).fill(initial);
	const b = builder.constant(descB, bufferB);
	const c = builder.matmul(a, b);

	const graph = builder.build({c});
	const bufferA = new Float32Array(sizeOfShape(descA.dimensions)).fill(0.5);
	const bufferC = new Float32Array(sizeOfShape(dimRes));
	const inputs = {'a': bufferA};
	const outputs = {'c': bufferC};
	graph.compute(inputs, outputs);
	return bufferC
}

function sizeOfShape(shape){
	return shape.reduce((a,b)=>a*b)
}
