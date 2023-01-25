#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

struct Point
{
	int x;
	int y;
};

__constant__ double avg_dev[32][3];
__constant__ double _cov_dev[32][3][3];

__global__ void kernel(uchar4 *data, int h, int w, int nc)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	int n = w * h;
	int i, j, k;
	double sum, max_sum;
	int res = 0;
	uchar4 *p;
	double tmp[3];

	// if (idx == 0){
	// 	for (i = 0; i < nc; i++)
	// 		// for (j = 0; j < 3; j++)
	// 			printf("GPU [%d]: %lf, %lf, %lf\n", i, avg_dev[i][0], avg_dev[i][1], avg_dev[i][2]);
	// }

	while (idx < n)
	{
		p = &data[idx];
		for (i = 0; i < nc; i++)
		{
			double sub[3] = {0};
			sub[0] = p->x - avg_dev[i][0];
			sub[1] = p->y - avg_dev[i][1];
			sub[2] = p->z - avg_dev[i][2];

			for (j = 0; j < 3; j++)
			{
				tmp[j] = 0.0;
				for (k = 0; k < 3; k++)
				{
					tmp[j] = tmp[j] + sub[k] * _cov_dev[i][k][j];
				}
			}
			sum = 0.0;
			for (k = 0; k < 3; k++)
			{
				sum = sum + tmp[k] * sub[k];
			}
			sum = -1 * sum;
			if (i == 0)
			{
				max_sum = sum;
				res = i;
			}
			else
			{
				if (sum > max_sum)
				{
					max_sum = sum;
					res = i;
				}
			}
		}
		p->w = (unsigned char)res;
		idx += offset;
	}
}

int main()
{
	int w, h, i, j, k, nc, np;

	// Read params
	char fin[256], fout[256];
	scanf("%s", fin);
	scanf("%s", fout);
	scanf("%d", &nc);

	std::vector<std::vector<Point> > points(nc);
	for (i = 0; i < nc; i++)
	{
		scanf("%d", &np);
		points[i].resize(np);
		for (j = 0; j < np; j++)
		{
			scanf("%d", &points[i][j].x);
			scanf("%d", &points[i][j].y);
		}
	}

	// printf("%s\n", fin);
	// printf("%s\n", fout);
	// printf("%d\n", nc);

	// for (i = 0; i < nc; i++)
	// {
	// 	for (j = 0; j < points[i].size(); j++)
	// 	{
	// 		printf("%d %d\n", points[i][j].x,  points[i][j].y);
	// 	}
	// }

	// Read file
	FILE *fp = fopen(fin, "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	// Compute AVG COV
	double avg[nc][3];
	double cov[nc][3][3];
	for (i = 0; i < nc; i++)
	{
		for (j = 0; j < 3; j++)
		{
			avg[i][j] = 0.0;
		}
	}
	for (i = 0; i < nc; i++)
	{
		for (j = 0; j < 3; j++)
		{
			for (k = 0; j < 3; j++)
			{
				cov[i][j][k] = 0.0;
			}
		}
	}
	for (i = 0; i < nc; i++)
	{
		np = points[i].size();

		for (j = 0; j < np; j++)
		{
			uchar4 p = data[points[i][j].y * w + points[i][j].x];
			avg[i][0] = avg[i][0] + p.x;
			avg[i][1] = avg[i][1] + p.y;
			avg[i][2] = avg[i][2] + p.z;
		}

		avg[i][0] = avg[i][0] / np;
		avg[i][1] = avg[i][1] / np;
		avg[i][2] = avg[i][2] / np;
	}

	for (i = 0; i < nc; i++)
	{
		np = points[i].size();

		for (j = 0; j < np; j++)
		{
			uchar4 p = data[points[i][j].y * w + points[i][j].x];
			double sub[3];

			sub[0] = p.x - avg[i][0];
			sub[1] = p.y - avg[i][1];
			sub[2] = p.z - avg[i][2];

			// printf("%lf %lf %lf", sub[0], sub[1], sub[2]);

			for (k = 0; k < 3; k++)
			{
				cov[i][k][0] = cov[i][k][0] + sub[k] * sub[0];
				cov[i][k][1] = cov[i][k][1] + sub[k] * sub[1];
				cov[i][k][2] = cov[i][k][2] + sub[k] * sub[2];
			}
		}

		for (k = 0; k < 3; k++)
		{
			cov[i][k][0] = cov[i][k][0] / (np - 1);
			cov[i][k][1] = cov[i][k][1] / (np - 1);
			cov[i][k][2] = cov[i][k][2] / (np - 1);
		}
	}

	// Calculate inv COV
	double D;
	double _cov[nc][3][3];
	for (i = 0; i < nc; i++)
	{
		D = (cov[i][0][0] * cov[i][1][1] * cov[i][2][2] + cov[i][1][0] * cov[i][2][1] * cov[i][0][2] + cov[i][0][1] * cov[i][1][2] * cov[i][2][0]) -
			(cov[i][0][2] * cov[i][1][1] * cov[i][2][0] + cov[i][2][1] * cov[i][1][2] * cov[i][0][0] + cov[i][0][1] * cov[i][1][0] * cov[i][2][2]);

		_cov[i][0][0] = (cov[i][1][1] * cov[i][2][2] - cov[i][2][1] * cov[i][1][2]) / D;
		_cov[i][0][1] = (cov[i][0][2] * cov[i][2][1] - cov[i][0][1] * cov[i][2][2]) / D;
		_cov[i][0][2] = (cov[i][0][1] * cov[i][1][2] - cov[i][0][2] * cov[i][1][1]) / D;
		_cov[i][1][0] = (cov[i][1][2] * cov[i][2][0] - cov[i][1][0] * cov[i][2][2]) / D;
		_cov[i][1][1] = (cov[i][0][0] * cov[i][2][2] - cov[i][0][2] * cov[i][2][0]) / D;
		_cov[i][1][2] = (cov[i][1][0] * cov[i][0][2] - cov[i][0][0] * cov[i][1][2]) / D;
		_cov[i][2][0] = (cov[i][1][0] * cov[i][2][1] - cov[i][2][0] * cov[i][1][1]) / D;
		_cov[i][2][1] = (cov[i][2][0] * cov[i][0][1] - cov[i][0][0] * cov[i][2][1]) / D;
		_cov[i][2][2] = (cov[i][0][0] * cov[i][1][1] - cov[i][1][0] * cov[i][0][1]) / D;
	}

	// for (i = 0; i < nc; i++)
	// 	for (j = 0; j < 3; j++)
	// 		printf("CPU [%d][%d]: %lf, %lf, %lf\n", i, j, _cov[i][j][0], _cov[i][j][1], _cov[i][j][2]);

	cudaMemcpyToSymbol(avg_dev, avg, sizeof(double) * nc * 3);
	cudaMemcpyToSymbol(_cov_dev, _cov, sizeof(double) * nc * 3 * 3);

	uchar4 *data_dev;
	cudaMalloc(&data_dev, sizeof(uchar4) * h * w);
	cudaMemcpy(data_dev, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice);
	kernel<<<256, 256>>>(data_dev, h, w, nc);
	cudaGetLastError();
	cudaMemcpy(data, data_dev, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost);

	// Write file
	fp = fopen(fout, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	return 0;
}
