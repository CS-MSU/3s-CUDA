#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                                                 \
	do                                                            \
	{                                                             \
		cudaError_t res = call;                                   \
		if (res != cudaSuccess)                                   \
		{                                                         \
			fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
					__FILE__, __LINE__, cudaGetErrorString(res)); \
			exit(0);                                              \
		}                                                         \
	} while (0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *out, int w, int h, int r)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int m, n;
	int c;
	uchar4 p;

	for (y = idy; y < h; y += offsety)
	{
		for (x = idx; x < w; x += offsetx)
		{
			if (r == 0)
			{
				p = tex2D(tex, x, y);
				out[y * w + x] = make_uchar4(p.x, p.y, p.z, p.w);
			}

			else
			{
				int red, green, blue;
				int dict_r[256] = {0};
				int dict_g[256] = {0};
				int dict_b[256] = {0};

				int counter = 0;
				for (m = -1 * r; m <= r; m++)
				{
					if ((y + m < 0) || (y + m >= h))
						continue;
					for (n = -1 * r; n <= r; n++)
					{
						if ((x + n < 0) || (x + n >= w))
							continue;
						p = tex2D(tex, x + n, y + m);
						dict_r[p.x]++;
						dict_g[p.y]++;
						dict_b[p.z]++;
						counter++;
					}
				}
				for (c = 1; c < 256; c++)
				{
					dict_b[c] += dict_b[c - 1];

					if (dict_b[c] >= (counter / 2 + 1))
					{
						blue = c;
						break;
					}
				}

				for (c = 1; c < 256; c++)
				{
					dict_r[c] += dict_r[c - 1];

					if (dict_r[c] >= (counter / 2 + 1))
					{
						red = c;
						break;
					}
				}

				for (c = 1; c < 256; c++)
				{
					dict_g[c] += dict_g[c - 1];

					if (dict_g[c] >= (counter / 2 + 1))
					{
						green = c;
						break;
					}
				}

				p = tex2D(tex, x, y);
				out[y * w + x] = make_uchar4(red, green, blue, p.w);
			}
		}
	}
}

int main()
{

	int w, h, r;
	char fin[256], fout[256];
	scanf("%s", fin);
	scanf("%s", fout);
	scanf("%d", &r);
	FILE *fp = fopen(fin, "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

	tex.normalized = false;
	tex.filterMode = cudaFilterModePoint;
	tex.channelDesc = ch;
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;

	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

	kernel<<<dim3(16, 16), dim3(32, 32)>>>(dev_out, w, h, r);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
	CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(fout, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}