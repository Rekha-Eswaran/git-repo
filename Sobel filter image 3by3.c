// Rekha
//Sobel filter 3 * 3 with vertical and horizontal filters using unesco
//Gx = horizontal filter to find vertical edges
//Gy = vertical filter to find horizontal edges

#include <stdio.h>
#include <math.h>

int main(void)
{

       const int Gx[3][3]= {{ -1,  0,  1 },           //horizontal filter
                           { -2,  0,  2 },
                           { -1,  0,  1 }};

      const int Gy[3][3]= {{ -1,  -2,  -1 },
                           { 0,  0,  0 },
                           { 1,  2,  1 }};             //vertical filter

    //unesco image
    unsigned char unesco_input_image[500][750];        //height of unesco image =500, width of unesco image= 750
    unsigned char unesco_Gx_output_image[500][750];
    unsigned char unesco_Gy_output_image[500][750];

    //read unesco raw image file
    FILE * fp = fopen( "unesco750-rawfile.raw", "rb" );      //open the input raw file

    fread(unesco_input_image,1,375000,fp);
    fclose (fp);


    //declared variables
    int  img_height,img_width,fil_height,fil_width, x, y, i, j, H, W;
    double x_sum, y_sum;                                             //vertical and horizontal sum of convolution


    //unesco image height and width
    img_height = 500;
    img_width = 750;


    //filter height and width
    fil_height = 3;
    fil_width = 3;

    H = floor((fil_height - 1)/2);
    W = floor((fil_width - 1)/2);


    //for loops for unesco image
      for (y = H; y < img_height-H; y++) {
        for (x = W; x < img_width-W; x++) {
          x_sum = 0.0;
          y_sum = 0.0;
          // iterate over the filter
          for (j = -H; j < H + 1; j++) {
            for ( i= -W; i < W + 1; i++) {
             //To avoid overflow, divide by greater than 4 (sum of all positive numbers in filter matrix). Here I divided by 8 since "division by 4 gives black spots"
             //To normalize, add 128 that is 256/2
             //convolution - multiply and summation of image pixel values and filter matrix
             x_sum += ((unesco_input_image[y + j][x + i] * Gx[H + j][W + i]) / 8) + 128;            //horizontal filter
             y_sum += ((unesco_input_image[y + j][x + i] * Gy[H + j][W + i]) / 8) + 128;            //vertical filter
            }
          }
          unesco_Gx_output_image[y][x] = x_sum;
          unesco_Gy_output_image[y][x] = y_sum;
        }
    }


    //write horizontal filtered image
    FILE * fp2 = fopen( "Gx_unesco_3by3_result_raw_8.raw", "wb" );
    fwrite( unesco_Gx_output_image, 1, 375000, fp2 );
    fclose (fp2);

    //write vertical filtered image
    FILE * fp3 = fopen( "Gy_unesco_3by3_result_raw_8.raw", "wb" );
    fwrite( unesco_Gy_output_image, 1, 375000, fp3 );
    fclose (fp3);


printf("Sobel 3by3 filter for unesco done!");
return 0;
}
