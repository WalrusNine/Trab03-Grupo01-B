/*
**	Disciplina:	SSC-0143 Programacao Concorrente
**
**	Docente:	Prof. Julio Cezar Estrella
**
**	Discentes:	Andre Miguel Coelho Leite	8626249
**			Laerte Vidal Junior		7557800
**
**	Trabalho 3:	Smooth com CUDA
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define RGB 0
#define RBG 1
#define GRB 2
#define GBR 3
#define BRG 4 
#define BGR 5

/*
**	Image is stored in an array of PIXELs,
**	which have red, green and blue values
**	as unsigned chars (RGB) or just one
**	unsigned char(GRS).
**
**
**	SMOOTH
**	5x5
**	(i-2,j-2)	(i-1,j-2)	(i  ,j-2)	(i+1,j-2)	(i+2,j-2)
**	(i-2,j-1)	(i-1,j-1)	(i  ,j-1)	(i+1,j-1)	(i+2,j-1)
**	(i-2,  j)	(i-1,  j)	(i  ,  j)	(i+1,  j)	(i+2,  j)
**	(i-2,j+1)	(i-1,j+1)	(i  ,j+1)	(i+1,j+1)	(i+2,j+1)
**	(i-2,j+2)	(i-1,j+2)	(i  ,j+2)	(i+1,j+2)	(i+2,j+2)
*/

/* RGB PIXEL */
typedef struct{
	unsigned char r, g, b;
}PIXELRGB;

/* GRAYSCALE PIXEL */
typedef struct{
	unsigned char i;
}PIXELGRS;

/* UNION for keeping either grayscale or rgb PIXEL */
typedef union{
	PIXELGRS grs;
	PIXELRGB rgb;
} PIXEL;

typedef struct image{
	PIXEL* pixel;
	int width, height;
}IMAGE;

/* Prototypes */
IMAGE* read_ppm_image();
void write_ppm(const char *,IMAGE*,int);
void delete_image(IMAGE**);

__global__ void smooth_grs(PIXEL*,PIXEL*, int, int);
__global__ void smooth_rgb(PIXEL*,PIXEL*, int, int);

int timeval_subtract(struct timeval*, struct timeval*, struct timeval*);

/* Globals */
int grayscale = 0;

int main(int argc, char** argv)
{
	IMAGE* 	image;
	PIXEL*	gpixels;

	/* Time variables */
	struct timeval t_begin, t_end, t_diff;

	/* Read image */	
	image = read_ppm_image();
	gpixels = image->pixel;

	/* Get time start */
	gettimeofday(&t_begin, NULL);

	/* Size */
	size_t size = (image->width)*(image->height)*sizeof(PIXEL);

	/* Device arrays */
	PIXEL* d_pixels_in;
	cudaMalloc(&d_pixels_in, size);
	cudaMemcpy(d_pixels_in, gpixels, size, cudaMemcpyHostToDevice);

	PIXEL* d_pixels_out;
	cudaMalloc(&d_pixels_out, size);

	/* Setup blocks and threads */
	dim3 threadsPerBlock( 32, 32 );
	dim3 numBlocks( (image->width) / threadsPerBlock.x, (image->height) / threadsPerBlock.y );
	
	/* Run smooth */
	if (grayscale) smooth_grs<<<numBlocks, threadsPerBlock>>>(d_pixels_in, d_pixels_out, image->width, image->height);
	else smooth_rgb<<<numBlocks, threadsPerBlock>>>(d_pixels_in, d_pixels_out, image->width, image->height);

	/* Get time end */
	gettimeofday(&t_end, NULL);

	/* Get diff time and print in stderr */
	timeval_subtract(&t_diff, &t_end, &t_begin);
	fprintf(stderr, "%ld.%06ld\n", t_diff.tv_sec, t_diff.tv_usec);

	/* Copy results */
	cudaMemcpy(image->pixel, d_pixels_out, size, cudaMemcpyDeviceToHost);

	/* Write resulting image */
	write_ppm("out.ppm",image,RGB);

	/* Free memory */
	delete_image(&image);

	cudaFree(d_pixels_in);
	cudaFree(d_pixels_out);
	
	return EXIT_SUCCESS;
}

	
IMAGE* read_ppm_image()
{
	/*
	**	Snippet adapted from:
	**	http://stackoverflow.com/questions/2693631/read-ppm-file-and-store-it-in-an-array-coded-with-c
	*/
	
	FILE* fp = fopen("in.ppm", "rb");
	
	/*
	**	PX
	**	n_columns m_rows
	**	max_color
	**	row_1 -> column_1  ...  column_n
	**	...
	**	row_n -> column_1  ...  column_n
	**
	**	each row is in format (255 255 255) => (R G B)
	*/

	/* Get Type PPM */
	char type[3];
	fscanf(fp, "%s\n", type);
	if (type[1] == '5') grayscale = 1;
	
	/* Check for comments */
	char c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}
	ungetc(c, fp);

	/* Get Size */
	int width, height;
	fscanf(fp, "%d %d\n", &width, &height);

	/* Get Max Color */
	fscanf(fp, "%*d\n");

	/* Create Image */
	IMAGE* image	= (IMAGE*)malloc(sizeof(IMAGE));
	image->width 	= width;
	image->height 	= height;
	image->pixel 	= (PIXEL *)malloc(width*height*sizeof(PIXEL));

	/* Read image's pixel data */
	if (grayscale){
		int i, j;
		for (i = 0; i < height; ++i){
			for (j = 0; j < width; ++j){
				fread(&(image->pixel[i*width + j].grs.i), 1, 1, fp);
			}
		}
	}
	else fread(image->pixel, sizeof(PIXEL),width*height, fp);

	fclose(fp);

	return image;
}

__global__ void smooth_rgb(PIXEL* in, PIXEL* out, int width, int height){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( i >= width || j >= height )
		return;

	int k, l;
	int red, green, blue;
	red = green = blue = 0;
	for (k = -2; k <= 2; k++){
		for (l = -2; l <= 2; l++){
			if (i+k >= 0 && i+k < width && j+l >= 0 && j+l < height){
				red 	+= in[(j+l)*width + (i+k)].rgb.r;
				green 	+= in[(j+l)*width + (i+k)].rgb.g;
				blue 	+= in[(j+l)*width + (i+k)].rgb.b;
			}
		}
	}
	out[(j)*width + (i)].rgb.r = (red / 25);
	out[(j)*width + (i)].rgb.g = (green / 25);
	out[(j)*width + (i)].rgb.b = (blue / 25);
}

__global__ void smooth_grs(PIXEL* in, PIXEL* out, int width, int height){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( i >= width || j >= height )
		return;

	int k, l;
	int mean = 0;
	for (k = -2; k <= 2; k++){
		for (l = -2; l <= 2; l++){
			if (i+k >= 0 && i+k < width && j+l >= 0 && j+l < height){
				mean += in[(j+l)*width + (i+k)].grs.i;
			}
		}
	}
	out[(j)*width + (i)].grs.i = (mean / 25);
}

void write_ppm(const char *fname,IMAGE* image,int m){
	FILE *fp = fopen(fname, "wb");
	
	/* Put header */
	if (grayscale) fprintf(fp, "P5\n");
	else fprintf(fp, "P6\n");

	/* Put size */
	fprintf(fp, "%d %d\n",image->width, image->height);
	int width = image->width, height = image->height;

	/* RGB component depth */
	fprintf(fp, "%d\n", 255);

	/* Pixel data */
	int i, j;
	for (j = 0; j < height; j++){
		for (i = 0; i < width; i++){
			if (grayscale){
				fwrite(&(((image->pixel)[j*width + i]).grs.i), sizeof(unsigned char),1, fp);
			} else{
				fwrite(&(image->pixel[j*width + i].rgb.r), sizeof(unsigned char),1, fp);
				fwrite(&(image->pixel[j*width + i].rgb.g), sizeof(unsigned char),1, fp);
				fwrite(&(image->pixel[j*width + i].rgb.b), sizeof(unsigned char),1, fp);
			}
		}
	}
	
	fclose(fp);
}

void delete_image(IMAGE** image){
	if (*image != NULL){
		free((*image)->pixel);
		free(*image);
		*image = NULL;
	}
}


/*
**	Snippet from:
**	http://stackoverflow.com/questions/1468596/calculating-elapsed-time-in-a-c-program-in-milliseconds
*/

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}
