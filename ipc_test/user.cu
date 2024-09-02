//app 2, part of a 2-part IPC example                                                                                                          
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 3

#define cudaCheckErrors(msg) \
  do { \
  cudaError_t __err = cudaGetLastError(); \
  if (__err != cudaSuccess) { \
  fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
    msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
  fprintf(stderr, "*** FAILED - ABORTING\n"); \
  exit(1); \
  } \
  } while (0)

int main(){
  float *data;
  float h_nums[DSIZE];

  cudaIpcMemHandle_t my_handle;
  unsigned char handle_buffer[sizeof(my_handle)+1];
  memset(handle_buffer, 0, sizeof(my_handle)+1);
  FILE *fp;
  fp = fopen("testfifo", "r");
  if (fp == NULL) {printf("fifo open fail \n"); return 1;}
  int ret;
  for (int i = 0; i < sizeof(my_handle); i++){
    ret = fscanf(fp,"%c", handle_buffer+i);
    if (ret == EOF) printf("received EOF\n");
    else if (ret != 1) printf("fscanf returned %d\n", ret);}
  memcpy((unsigned char *)(&my_handle), handle_buffer, sizeof(my_handle));
  cudaIpcOpenMemHandle((void **)&data, my_handle, cudaIpcMemLazyEnablePeerAccess);
  cudaCheckErrors("IPC handle fail");
  int flag=0;
  while (flag==0){
    //cudaMemset(data, 1, sizeof(float));                                                                                                        
    cudaMemcpy(h_nums, data, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaCheckErrors("memset fail");                                                                                                            
    cudaCheckErrors("memcopy fail");
    printf("values read from GPU memory : %f %f %f\n", h_nums[0], h_nums[1], h_nums[2]);
    sleep(1);
    if (h_nums[0]<0){
        flag=-1;
    }
  }
  cudaIpcCloseMemHandle(&data);
  cudaFree(&data);
  return 0;
}