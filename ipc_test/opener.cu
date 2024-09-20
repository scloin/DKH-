//app 1, part of a 2-part IPC example                                                                                                                                                                                                                                                                                    
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 28*28

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

__global__ void polling(int* head){
  int check=1;
  while(check!=0){
    //wait for head==0
    check=atomicExch(&head[0],1);
  }
}

__global__ void set_tail(int* tail){
  atomicExch(&tail[0],0);
}

int main(){
  system("rm -f testfifo"); // remove any debris                                                                                                                                                                                                                                                                         
  int ret = mkfifo("testfifo", 0600); // create fifo                                                                                                                                                                                                                                                                     
  if (ret != 0) {printf("mkfifo error: %d\n",ret); return 1;}


  float h_nums[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float h_nums2[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float h_nums3[] = {-1};
  void *data;
  cudaMalloc(&data, DSIZE*sizeof(float)+3*sizeof(int));
  int* out = (int *)((float*)data+DSIZE);
  int* head=(int *)((float*)data+DSIZE)+1;
  int* tail=(int *)((float*)data+DSIZE)+2;

  int h_out[]={9707, -1, -1};
  int hhead[]={9};
  cudaIpcMemHandle_t my_handle;

  cudaMemcpy(out, h_out, 3*sizeof(int), cudaMemcpyHostToDevice);

  cudaCheckErrors("malloc fail");                                                                                                                                                                                                                                                        
  cudaIpcGetMemHandle(&my_handle, data);
  unsigned char handle_buffer[sizeof(my_handle)+1];
  memset(handle_buffer, 0, sizeof(my_handle)+1);
  memcpy(handle_buffer, (unsigned char *)(&my_handle), sizeof(my_handle));
  cudaCheckErrors("get IPC handle fail");
  FILE *fp;
  printf("waiting for app2\n");
  fp = fopen("testfifo", "w");
  if (fp == NULL) {printf("fifo open fail \n"); return 1;}
  printf("%ld\n", sizeof(my_handle));
  for (int i=0; i < sizeof(my_handle); i++){
    ret = fprintf(fp,"%c", handle_buffer[i]);
    if (ret != 1) printf("ret = %d\n", ret);}
  fclose(fp);


  sleep(2); // wait for app 2 to modify data                                                                                                                                                                                                                                                                             
  //float *result = (float *)malloc(DSIZE*sizeof(float));
  
  // cudaMemcpy(result, data, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  printf("7\n");
  cudaMemcpy((float*)data, h_nums, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  set_tail<<<1,1>>>(tail); 
  polling<<<1,1>>>(head);    
  cudaDeviceSynchronize();
  printf("1\n");
  sleep(2);
  cudaMemcpy((float*)data, h_nums2, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  set_tail<<<1,1>>>(tail);
  polling<<<1,1>>>(head); 
  cudaDeviceSynchronize();
  printf("7\n");
  sleep(2);
  cudaMemcpy((float*)data, h_nums, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  set_tail<<<1,1>>>(tail); 
  polling<<<1,1>>>(head);    
  cudaDeviceSynchronize();
  printf("1\n");
  sleep(2);
  cudaMemcpy((float*)data, h_nums2, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  set_tail<<<1,1>>>(tail);
  polling<<<1,1>>>(head); 
  cudaDeviceSynchronize();
  printf("7\n");
  sleep(2);
  cudaMemcpy((float*)data, h_nums, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  set_tail<<<1,1>>>(tail); 
  polling<<<1,1>>>(head);    
  cudaDeviceSynchronize();
  printf("1\n");
  sleep(2);
  cudaMemcpy((float*)data, h_nums2, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  set_tail<<<1,1>>>(tail);
  polling<<<1,1>>>(head); 
  cudaDeviceSynchronize();
  printf("e\n");
  sleep(2);
  cudaMemcpy((float*)data, h_nums3, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(result, (float*)data, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);


  //printf("%lf\n", *result);
  // if (!(*result)) printf("Fail!\n");
  // else printf("Success!\n");
  system("rm testfifo");
  cudaDeviceReset();
  return 0;
}