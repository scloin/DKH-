//app 1, part of a 2-part IPC example                                                                                                                                                                                                                                                                                    
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
  system("rm -f testfifo"); // remove any debris                                                                                                                                                                                                                                                                         
  int ret = mkfifo("testfifo", 0600); // create fifo                                                                                                                                                                                                                                                                     
  if (ret != 0) {printf("mkfifo error: %d\n",ret); return 1;}

  float h_nums[] = {1.1111, 2.2222, 3.141592654};
  float h_nums2[] = {1, 2.212342, 2.34};
  float h_nums3[] = {-1, 2.212342, 2.34};
  float *data;
  cudaIpcMemHandle_t my_handle;
  cudaMalloc(&data, DSIZE*sizeof(float));
  cudaCheckErrors("malloc fail");
  //cudaMemset(data, 0, DSIZE*sizeof(int));                                                                                                                                                                                                                                                                              
  //cudaCheckErrors("memset fail");                                                                                                                                                                                                                                                                                      
  cudaMemcpy(data, h_nums, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("memcoy fail");
  cudaIpcGetMemHandle(&my_handle, data);
  unsigned char handle_buffer[sizeof(my_handle)+1];
  memset(handle_buffer, 0, sizeof(my_handle)+1);
  memcpy(handle_buffer, (unsigned char *)(&my_handle), sizeof(my_handle));
  cudaCheckErrors("get IPC handle fail");
  FILE *fp;
  printf("waiting for app2\n");
  fp = fopen("testfifo", "w");
  if (fp == NULL) {printf("fifo open fail \n"); return 1;}
  printf("%d\n", sizeof(my_handle));
  for (int i=0; i < sizeof(my_handle); i++){
    ret = fprintf(fp,"%c", handle_buffer[i]);
    if (ret != 1) printf("ret = %d\n", ret);}
  fclose(fp);
  sleep(2); // wait for app 2 to modify data                                                                                                                                                                                                                                                                             
  float *result = (float *)malloc(DSIZE*sizeof(float));
  
  cudaMemcpy(result, data, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data, h_nums2, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  sleep(5);
  cudaMemcpy(data, h_nums3, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  if (!(*result)) printf("Fail!\n");
  else printf("Success!\n");
  system("rm testfifo");
  return 0;
}