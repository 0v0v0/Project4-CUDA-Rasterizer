/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "device_functions.h"
//#include <sm_32_atomic_functions.h>
//#include <device_atomic_functions.h>
//#include "sm_32_atomic_functions.hpp"
//#include <sm_20_atomic_functions.h>

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation

		glm::vec3 color;
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;

		// ...
		//TextureData* dev_diffuseTex;
		//int texWidth, texHeight;
		
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int* depth) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {

		if (depth[index] != INT_MAX)
		{
			// TODO: add your fragment shader code here
			float cosinepower = glm::dot(glm::vec3(0, 0, 1), fragmentBuffer[index].eyeNor) / glm::length(fragmentBuffer[index].eyeNor);
			framebuffer[index] = fragmentBuffer[index].color*cosinepower;
		}
		else
		{
			//If point is in void, paint mono color
			//PINK= 1 0.07 0.5
			framebuffer[index] = glm::vec3(0.3f,0.3f,0.3f);
		}	
    }
}

__global__
void render2(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int* depth) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {

		if (depth[index] != INT_MAX)
		{
			// TODO: add your fragment shader code here
			framebuffer[index]=fragmentBuffer[index].color;
		}
		else
		{
			//If point is in void, paint mono color
			//PINK= 1 0.07 0.5
			framebuffer[index] = glm::vec3(0.3f, 0.3f, 0.3f);
		}
	}
}


/**
* Writes fragment colors to the framebuffer
*/
__global__
void toon_shader(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int* depth, float layers) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {

		if (depth[index] != INT_MAX)
		{
			// TODO: add your fragment shader code here
			
			/*
			************
			Toon Shader: 
			************
			1. divide color space into 5 segments, and cartoonrize the color
			2. draw outline when: 
				Red wires: depth drops exceeds the threshold (Replace the normal checking, so we could control the wire width) 
				Green wires: normal differs exceeds the threshold
			*/

			//make the color spartial
			glm::vec3 color = fragmentBuffer[index].color;
			int r = color.x * layers;
			int g = color.y * layers;
			int b = color.z * layers;
			r /= 2;
			b /= 2;
			g /= 2;
			framebuffer[index] = glm::vec3(r,g,b)*(1.0f/layers);

			//draw outline

			//threshold for depth, bigger=stronger wires
			float threshold = 100;

			//threshold for normal, bigger=stronger wires
			float norm_threshold = 0.8f;

			//4 directions to check
			int up = index = x + ((y - 1) * w);
			int down = index = x + ((y + 1) * w);
			int left = index = x - 1 + (y * w);
			int right = index = x + 1 + (y * w);

			//differs in depth
			int dif_up, dif_down, dif_left, dif_right;
			//differs in normal
			float norm_left, norm_right, norm_up, norm_down;

			//Init
			dif_up = dif_down = dif_left = dif_right = 0;
			norm_left = norm_right = norm_up = norm_down = 0;

			//Compute differs
			if (up > 0 && up < w*h)
			{
				dif_up = abs(depth[up] - depth[index]);
				norm_up = glm::dot(fragmentBuffer[up].eyeNor, fragmentBuffer[index].eyeNor);
			}
			if (down > 0 && down < w*h)
			{
				dif_down = abs(depth[down] - depth[index]);
				norm_down = glm::dot(fragmentBuffer[down].eyeNor, fragmentBuffer[index].eyeNor);
			}
			if (left > 0 && left < w*h)
			{
				dif_left = abs(depth[left] - depth[index]);
				norm_left = glm::dot(fragmentBuffer[left].eyeNor, fragmentBuffer[index].eyeNor);
			}
			if (right > 0 && right < w*h)
			{
				dif_right = abs(depth[right] - depth[index]);
				norm_right = glm::dot(fragmentBuffer[right].eyeNor, fragmentBuffer[index].eyeNor);
			}

			//Draw wires

			//Uncomment this For debugging 
			/*
			if (dif_up > threshold || dif_down > threshold || dif_left > threshold || dif_right > threshold)
			{
				framebuffer[index] = glm::vec3(1, 0, 0); //red wires
			}
			else if ( norm_left < norm_threshold || norm_right < norm_threshold || norm_up < norm_threshold || norm_down < norm_threshold)
			{				
				framebuffer[index] = glm::vec3(0, 1, 0); //green wires				
			}
			*/

			//Uncomment this For picture		
			if (norm_left < norm_threshold || norm_right < norm_threshold || norm_up < norm_threshold || norm_down < norm_threshold)
			{
				framebuffer[index] *= glm::vec3(0.3f); //gray wires				
			}
			if (dif_up > threshold || dif_down > threshold || dif_left > threshold || dif_right > threshold)
			{
				framebuffer[index] *= glm::vec3(0); //black wires
			}	
			
		}
		else
		{
			//If point is in void, paint mono color
			//PINK= 1 0.07 0.5
			framebuffer[index] = glm::vec3(1);
		}
	}
}

__global__
void toon_shader2(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int* depth, float layers, float threshold) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {

		if (depth[index] != INT_MAX)
		{
			//make the color spartial
			glm::vec3 color = fragmentBuffer[index].color;

			if (color.x < threshold)
			{				
				//density=x^2
				
				float fr = (1-color.x) * layers;
				float fg = (1-color.y) * layers;
				float fb = (1-color.z) * layers;			

				int r = fr;
				int g = fg;
				int b = fb;

				r %= 2;
				b %= 2;
				g %= 2;

				framebuffer[index] = glm::vec3(r*color.x, g*color.y, b*color.z);
			}
			else
			{
				framebuffer[index] = glm::vec3(1);
			}
		}
		else
		{
			//If point is in void, paint mono color
			//PINK= 1 0.07 0.5
			framebuffer[index] = glm::vec3(1);
		}
	}
}


//Sketch Shader
__global__
void sketch_shader(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int* depth, float threshold) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {

		if (depth[index] != INT_MAX)
		{

			//4 directions to check
			int up = index = x + ((y - 1) * w);
			int down = index = x + ((y + 1) * w);
			int left = index = x - 1 + (y * w);
			int right = index = x + 1 + (y * w);

			//differs in depth
			float dif_up, dif_down, dif_left, dif_right;
			//differs in normal
			float norm_left, norm_right, norm_up, norm_down;

			//Init
			dif_up = dif_down = dif_left = dif_right = 0;
			norm_left = norm_right = norm_up = norm_down = 0;

			//Compute differs
			if (up > 0 && up < w*h)
			{
				dif_up = glm::length(framebuffer[up] - framebuffer[index]);
			}
			if (down > 0 && down < w*h)
			{
				dif_down = glm::length(framebuffer[down] - framebuffer[index]);			
			}
			if (left > 0 && left < w*h)
			{
				dif_left = glm::length(framebuffer[left] - framebuffer[index]);				
			}
			if (right > 0 && right < w*h)
			{
				dif_right = glm::length(framebuffer[right] - framebuffer[index]);			
			}	

			if (dif_up > threshold || dif_down > threshold || dif_left > threshold || dif_right > threshold)
			{
				framebuffer[index] = glm::vec3(0);
			}
			else
			{
				framebuffer[index] = glm::vec3(1);
			}

		}
		else
		{
			//If point is in void, paint mono color
			//PINK= 1 0.07 0.5
			//framebuffer[index] = glm::vec3(1);
		}
	}
}

//Sketch Shader
__global__
void sketch_shader2(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int* depth, float threshold_g) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {

		if (depth[index] != INT_MAX)
		{

			//4 directions to check
			int up = index = x + ((y - 1) * w);
			int down = index = x + ((y + 1) * w);
			int left = index = x - 1 + (y * w);
			int right = index = x + 1 + (y * w);

			//differs in depth
			float dif_up, dif_down, dif_left, dif_right;
			//differs in normal
			float norm_left, norm_right, norm_up, norm_down;

			//Init
			dif_up = dif_down = dif_left = dif_right = 0;
			norm_left = norm_right = norm_up = norm_down = 0;

			//Compute differs
			if (up > 0 && up < w*h)
			{
				dif_up = framebuffer[up].x - framebuffer[index].x;
			}
			if (down > 0 && down < w*h)
			{
				dif_down = framebuffer[down].x - framebuffer[index].x;
			}
			if (left > 0 && left < w*h)
			{
				dif_left = framebuffer[left].x - framebuffer[index].x;
			}
			if (right > 0 && right < w*h)
			{
				dif_right = framebuffer[right].x - framebuffer[index].x;
			}

			float threshold =threshold_g* framebuffer[index].x;

			if (dif_up > threshold || dif_down > threshold || dif_left > threshold || dif_right > threshold)
			{
				framebuffer[index] = glm::vec3(0);
			}
			else
			{
				framebuffer[index] = glm::vec3(1);
			}

		}
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array

		glm::vec3 tmp;

		tmp = primitive.dev_position[vid];

		glm::vec4 aaa(tmp.x, tmp.y, tmp.z, 1);

		glm::vec4 point = MVP*aaa;

		point /= point.w;
		point.x += 1;
		point.y = 1-point.y;

		point.x *= width/2;
		point.y *= height/2;
		point.z *= 10000;
		primitive.dev_verticesOut[vid].pos = point; //Position

		primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex; //Copy texture pointer
		primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
		primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;	
		primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
		primitive.dev_verticesOut[vid].eyeNor = MV_normal*primitive.dev_normal[vid];
		//(primitive.dev_verticesOut + vid)->color=primitive.
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;

			int a = pid + curPrimitiveBeginId;
			int b = iid % (int)primitive.primitiveType;
			int c = primitive.dev_indices[iid];

			dev_primitives[a].v[b]= primitive.dev_verticesOut[c];
			//COPY texture data

			
			dev_primitives[a].v[b].dev_diffuseTex
				= primitive.dev_verticesOut[c].dev_diffuseTex;
			dev_primitives[a].v[b].texcoord0= primitive.dev_verticesOut[c].texcoord0;
			dev_primitives[a].v[b].texHeight
				= primitive.dev_verticesOut[c].texHeight;
			dev_primitives[a].v[b].texWidth
				= primitive.dev_verticesOut[c].texWidth;
			dev_primitives[a].v[b].eyeNor
				= primitive.dev_verticesOut[c].eyeNor;
				

		}
		// TODO: other primitive types (point, line)
	}
	
}

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b) 

/*
//Raster via triangles
__global__ void raster_naive(
	int numPrim, 
	Primitive* dev_primitives, 
	
	int* dev_depth, 
	Fragment* fragments , 
	int width, int height)
{
	// index id
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int i, j;

	if (idx < numPrim)
	{
		glm::vec4 a = (dev_primitives + idx)->v[0].pos;
		glm::vec4 b = (dev_primitives + idx)->v[1].pos;
		glm::vec4 c = (dev_primitives + idx)->v[2].pos;		

		//get bounding box
		float xmax = MAX(MAX(a.x, b.x), c.x);
		float xmin = MIN(MIN(a.x, b.x), c.x);

		float ymax = MAX(MAX(a.y, b.y), c.y);
		float ymin = MIN(MIN(a.y, b.y), c.y);

		//compute only once per-triangle
		glm::vec3 ab = glm::vec3(a) - glm::vec3(b);
		glm::vec3 bc = glm::vec3(b) - glm::vec3(c);
		glm::vec3 ca = glm::vec3(c) - glm::vec3(a);
		ab.z = bc.z = ca.z = FLT_MIN;	//I think this should be 0, for safety, I give it a smallest number
		float abc = glm::length(glm::cross(ab, -ca));	//area of the triangle


		if (abc > 0)
		{		
			//Loop around pixels within bounding box
			for (j = ymin; j < ymax; j++)
			{
				for (i = xmin; i < xmax; i++)
				{
					int pixel = j*width + i;

					if (pixel < width*height)
					{
						
						//Get Help From: 
						//http://blackpawn.com/texts/pointinpoly/
						//2017-10-16 All right, I see these functions are included in header files. :<

						// Compute vectors        
						glm::vec3 v0 = ca;
						glm::vec3 v1 = -ab;
						glm::vec3 v2 = glm::vec3(i, j, FLT_MIN) - glm::vec3(a);

						// Compute dot products
						float dot00 = glm::dot(v0, v0);
						float dot01 = glm::dot(v0, v1);
						float dot02 = glm::dot(v0, v2);
						float dot11 = glm::dot(v1, v1);
						float dot12 = glm::dot(v1, v2);

						// Compute barycentric coordinates
						float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
						float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
						float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

						float xc = u;
						float xb = v;
						float xa = 1 - u - v;

						float pixel_z = a.z*xa + b.z*xb + c.z*xc;	//z-depth

						//Anyway, we need barycentric interpolation in this branch, 
						//so use areas to determine if the point is within the triangle.

						if((u >= 0) && (v >= 0) && (u + v < 1))
						{
							if (dev_depth[pixel] > pixel_z)	//If the depth is closer to camera
							{
								glm::vec2 tex_cord = (dev_primitives + idx)->v[0].texcoord0 *xa +
									(dev_primitives + idx)->v[1].texcoord0 *xb +
									(dev_primitives + idx)->v[2].texcoord0 *xc;	//texture_coordinate

								tex_cord = glm::normalize(tex_cord);

								
								if((dev_primitives + idx)->v[0].dev_diffuseTex !=NULL) //If has texture, paint
								{
									int theight = (dev_primitives + idx)->v[0].texHeight;
									int twidth = (dev_primitives + idx)->v[0].texWidth;

									float tx = tex_cord.x*twidth;
									float ty = tex_cord.y*theight;
									int tpos;
									tpos = (ty*twidth + tx);
									tpos *= 3;

									fragments[pixel].color = glm::vec3(
										((dev_primitives + idx)->v[0].dev_diffuseTex[tpos]),
										((dev_primitives + idx)->v[0].dev_diffuseTex[tpos + 1]),
										((dev_primitives + idx)->v[0].dev_diffuseTex[tpos + 2])
									);
								}
								else //No texture, paint white
								{
									fragments[pixel].color = glm::vec3(1, 1, 1);	//paint white for checking
								}

								//int atomicMin(int* address, int val);

								atomicMin(dev_depth+pixel, pixel_z);
								
								//dev_depth[pixel] = pixel_z; //replace the depth-buffer
							}
						}
					}
				}
			}
		}
	}
}
*/


//Raster via triangles
__global__ void raster_naive2(
	int numPrim,
	Primitive* dev_primitives,

	int* dev_depth,
	Fragment* fragments,
	int width, int height)
{
	// index id
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int i, j;

	if (idx < numPrim)
	{
		glm::vec4 a = (dev_primitives + idx)->v[0].pos;
		glm::vec4 b = (dev_primitives + idx)->v[1].pos;
		glm::vec4 c = (dev_primitives + idx)->v[2].pos;

		//get bounding box
		float xmax = MAX(MAX(a.x, b.x), c.x);
		float xmin = MIN(MIN(a.x, b.x), c.x);
		float ymax = MAX(MAX(a.y, b.y), c.y);
		float ymin = MIN(MIN(a.y, b.y), c.y);

		//compute only once per-triangle		
		glm::vec3 tri[3];
		tri[0] = glm::vec3(a);
		tri[1] = glm::vec3(b);
		tri[2] = glm::vec3(c);

		float area = calculateSignedArea(tri);	//area of the triangle

		if (area > 0)
		{
			//Loop around pixels within bounding box
			for (j = ymin; j < ymax; j++)
			{
				for (i = xmin; i < xmax; i++)
				{
					int pixel = j*width + i;

					if (pixel < width*height)
					{
						glm::vec2 point(i, j);

						glm::vec3 bary = calculateBarycentricCoordinate(tri, point);

						int pixel_z = -getZAtCoordinate(bary, tri);	//z-depth

						//Anyway, we need barycentric interpolation in this branch, 
						//so use areas to determine if the point is within the triangle.
						if ((bary.x >= 0) && (bary.y >= 0) && (bary.z >=0))
						{
							if (atomicMin(dev_depth + pixel, pixel_z) >= pixel_z)	//If the depth is closer to camera
							{																
								glm::vec3 nor= (dev_primitives + idx)->v[0].eyeNor *bary.x +
									(dev_primitives + idx)->v[1].eyeNor *bary.y +
									(dev_primitives + idx)->v[2].eyeNor *bary.z;
								fragments[pixel].eyeNor = nor;
								
								if ((dev_primitives + idx)->v[0].dev_diffuseTex != NULL) //If has texture, paint
								{
									//texture_coordinate
									glm::vec2 tex_cord; 								
									tex_cord = dev_primitives[idx].v[0].texcoord0 *bary.x +
										dev_primitives[idx].v[1].texcoord0 *bary.y +
										dev_primitives[idx].v[2].texcoord0 *bary.z;

									float theight = (dev_primitives + idx)->v[0].texHeight;
									float twidth = (dev_primitives + idx)->v[0].texWidth;

									float tx = (tex_cord.x)*twidth;
									float ty = (tex_cord.y)*theight;
									int tpos;
									tpos = ty*twidth + tx;								
									glm::vec3 color;
									if ((tpos < twidth*theight) && (tpos>=0))
									{
										tpos *= 3;
										//For 24bit only
										float color_r = dev_primitives[idx].v[0].dev_diffuseTex[tpos] / 255.0f;
										float color_g = dev_primitives[idx].v[1].dev_diffuseTex[tpos + 1] / 255.0f;
										float color_b = dev_primitives[idx].v[2].dev_diffuseTex[tpos + 2] / 255.0f;
										color = glm::vec3(color_r, color_g, color_b);
									}
									fragments[pixel].color = color;								
								}									
								else //No texture, paint white
								{
									fragments[pixel].color = glm::vec3(1, 1, 1);	//paint white for checking									
								}							
							}
						}
					}
				}
			}
		}	
	}
}


//Raster via triangles
__global__ void raster_naive3(
	int numPrim,
	Primitive* dev_primitives,

	int* dev_depth,
	Fragment* fragments,
	int width, int height,
	glm::vec3 obj_color)
{
	// index id
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int i, j;

	if (idx < numPrim)
	{
		glm::vec4 a = (dev_primitives + idx)->v[0].pos;
		glm::vec4 b = (dev_primitives + idx)->v[1].pos;
		glm::vec4 c = (dev_primitives + idx)->v[2].pos;

		//get bounding box
		float xmax = MAX(MAX(a.x, b.x), c.x);
		float xmin = MIN(MIN(a.x, b.x), c.x);
		float ymax = MAX(MAX(a.y, b.y), c.y);
		float ymin = MIN(MIN(a.y, b.y), c.y);

		//compute only once per-triangle		
		glm::vec3 tri[3];
		tri[0] = glm::vec3(a);
		tri[1] = glm::vec3(b);
		tri[2] = glm::vec3(c);

		float area = calculateSignedArea(tri);	//area of the triangle

		if (area > 0)
		{
			//Loop around pixels within bounding box
			for (j = ymin; j < ymax; j++)
			{
				for (i = xmin; i < xmax; i++)
				{
					int pixel = j*width + i;

					if (pixel < width*height)
					{
						glm::vec2 point(i, j);

						glm::vec3 bary = calculateBarycentricCoordinate(tri, point);

						int pixel_z = -getZAtCoordinate(bary, tri);	//z-depth

																	//Anyway, we need barycentric interpolation in this branch, 
																	//so use areas to determine if the point is within the triangle.
						if ((bary.x >= 0) && (bary.y >= 0) && (bary.z >= 0))
						{
							if (atomicMin(dev_depth + pixel, pixel_z) >= pixel_z)	//If the depth is closer to camera
							{



								glm::vec3 nor = (dev_primitives + idx)->v[0].eyeNor *bary.x +
									(dev_primitives + idx)->v[1].eyeNor *bary.y +
									(dev_primitives + idx)->v[2].eyeNor *bary.z;

								fragments[pixel].eyeNor = nor;

								float cosinepower = glm::dot(glm::vec3(0, 0, 1), nor) / glm::length(nor);
								//paint mono color for checking	
								fragments[pixel].color = obj_color* cosinepower;
							}
						}
					}
				}
			}
		}
	}
}


#define toonshader 0
#define sketchshader 0
#define sketchshader_good 1


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	int numtris = 0;

	// Vertex Process & primitive assembly
	
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}	
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	
	// TODO: rasterize
	// Copy depthbuffer colors into framebuffer

	dim3 numPrims((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
   
	glm::vec3 color(1, 1, 0);
#if toonshader
	raster_naive3 << <numPrims, numThreadsPerBlock >> > (totalNumPrimitives, dev_primitives, dev_depth, dev_fragmentBuffer, width, height, color);
	toon_shader << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_depth, 10.0f);

#elif sketchshader
	raster_naive3 << <numPrims, numThreadsPerBlock >> > (totalNumPrimitives, dev_primitives, dev_depth, dev_fragmentBuffer, width, height, color);
	toon_shader << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_depth, 40.0f);
	sketch_shader << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_depth, 0.15f);
#elif sketchshader_good
	color = glm::vec3(1);
	raster_naive3 << <numPrims, numThreadsPerBlock >> > (totalNumPrimitives, dev_primitives, dev_depth, dev_fragmentBuffer, width, height, color);
	toon_shader2 << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_depth, 50.0f, 0.8f);
	sketch_shader2 << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_depth, 0.5f);

#else
	raster_naive2 << <numPrims, numThreadsPerBlock >> > (totalNumPrimitives, dev_primitives, dev_depth, dev_fragmentBuffer, width, height);
	render2 << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_depth);
#endif //textureshader
	

	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
