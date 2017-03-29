#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

// #define NUM_SPHERES 2
#define NUM_SPHERES 200
// #define BLOCK_SIZE 1
#define BLOCK_SIZE 256

struct Sphere {
  float posX, posY, posZ;
  float velX, velY, velZ;
  float radius;
  unsigned id;
};

float *d_posBuf, *d_velBuf, *d_radBuf, *d_velBuf_temp;
unsigned *d_idBuf;
size_t d_posBuf_size, d_velBuf_size, d_radBuf_size, d_idBuf_size;

__global__ void swapVelBufKernel(float *d_velBuf, float *d_velBuf_temp) {
  unsigned index, i;

  index = threadIdx.x + blockDim.x * blockIdx.x;

  if(index < NUM_SPHERES) {
    for(i = 0; i < 3; i++) {
      d_velBuf[index * 3 + i] = d_velBuf_temp[index * 3 + i];
    }
  }
}

__global__ void fillTempVelKernel(float *d_posBuf, float *d_velBuf,
  float *d_radBuf, float *d_velBuf_temp) {
  __shared__ float posBufLocal[BLOCK_SIZE * 3];
  __shared__ float velBufLocal[BLOCK_SIZE * 3];
  __shared__ float radBufLocal[BLOCK_SIZE];
  unsigned i, j, k, index, otherIndex; // otherIndex = index of other sphere
  float pos[3], vel[3], outVel[3], rad, mass; // outVel = output velocity
  float otherPos[3], otherVel[3], otherRad, otherMass; // data of other sphere
  float delta[3]; // Used to compare distance between sphere centers
  float distanceSquared, radiusSumSquared;
  bool didImpact;

  index = threadIdx.x + blockDim.x * blockIdx.x;

  for(i = 0; i < 3; i++) {
    pos[i] = d_posBuf[index * 3 + i];
    vel[i] = d_velBuf[index * 3 + i];
    outVel[i] = 0;
  }

  rad = d_radBuf[index];
  mass = rad * rad * rad;  

  didImpact = false;

  for(i = 0; i < gridDim.x; i++) {
    // For each block, load posBufLocal and velBufLocal
    posBufLocal[threadIdx.x * 3 + 0] = d_posBuf[blockDim.x * 3 * i +
      threadIdx.x * 3 + 0];
    posBufLocal[threadIdx.x * 3 + 1] = d_posBuf[blockDim.x * 3 * i +
      threadIdx.x * 3 + 1];
    posBufLocal[threadIdx.x * 3 + 2] = d_posBuf[blockDim.x * 3 * i +
      threadIdx.x * 3 + 2];

    velBufLocal[threadIdx.x * 3 + 0] = d_velBuf[blockDim.x * 3 * i +
      threadIdx.x * 3 + 0];
    velBufLocal[threadIdx.x * 3 + 1] = d_velBuf[blockDim.x * 3 * i +
      threadIdx.x * 3 + 1];
    velBufLocal[threadIdx.x * 3 + 2] = d_velBuf[blockDim.x * 3 * i +
      threadIdx.x * 3 + 2];

    radBufLocal[threadIdx.x] = d_radBuf[blockDim.x * i + threadIdx.x];

    __syncthreads();

    // Now all of the local data is filled

    // Check against each sphere in local data
    for(j = 0; j < blockDim.x; j++) {
      otherIndex = j + blockDim.x * i; // j = threadIdx.x, i = blockIdx.x
      // Check if the sphere data is not garbage data or the same data
      if((otherIndex < NUM_SPHERES) && (otherIndex != index)) {
        // Fill data of other sphere
        for(k = 0; k < 3; k++) {
          otherPos[k] = posBufLocal[j * 3 + k];
          otherVel[k] = velBufLocal[j * 3 + k];
        }

        otherRad = radBufLocal[j];

        // Check to see if they collide
      
        // Get difference in x, y, and z (0, 1, 2) for indexes
        for(k = 0; k < 3; k++) {
          delta[k] = pos[k] - otherPos[k];
        }

        // Calculate distanceSquared and radiusSumSquared
        distanceSquared = 0;
        radiusSumSquared = 0;

        for(k = 0; k < 3; k++) {
          distanceSquared = distanceSquared + delta[k] * delta[k];
        }
      
        radiusSumSquared = (rad + otherRad) * (rad + otherRad);

        if(distanceSquared < radiusSumSquared) {
          // We collide
          didImpact = true;

          // Calculate the mass of the other sphere
          otherMass = otherRad * otherRad * otherRad;

          for(k = 0; k < 3; k++) {
            outVel[k] = outVel[k] +
              (mass - otherMass) / (mass + otherMass) * vel[k] +
              (2 * otherMass) / (mass + otherMass) * otherVel[k];
          }
        }
      }
    }
    
    __syncthreads();
    // Now that all threads in this block have processed the local data
    // We can now move to the next set of local data
  }

  // Put the new velocity in outVel into the velocity buffer
  if(!didImpact) {
    // If the object did not impact with anything, continue with current
    // velocity
    for(i = 0; i < 3; i++) {
      outVel[i] = vel[i];
    }
  }

  for(i = 0; i < 3; i++) {
    d_velBuf_temp[index * 3 + i] = outVel[i];
  }
}

void updateVelocity() {
  unsigned numBlocks;

  numBlocks = NUM_SPHERES / BLOCK_SIZE;
  if(NUM_SPHERES % BLOCK_SIZE) numBlocks++; 

  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  // Fill the temporary velocity buffer
  fillTempVelKernel<<<dimGrid, dimBlock>>>(d_posBuf, d_velBuf, d_radBuf,
    d_velBuf_temp);
  // Copy from the temporary buffer to the real buffer
  swapVelBufKernel<<<dimGrid, dimBlock>>>(d_velBuf, d_velBuf_temp);
}

double time_diff(struct timeval x, struct timeval y) {
  double x_ms, y_ms, diff;

  x_ms = (double) x.tv_sec * 1000000 + (double) x.tv_usec;
  y_ms = (double) y.tv_sec * 1000000 + (double) y.tv_usec;

  diff = (double) y_ms - (double) x_ms;

  return diff;
}

void updateVelocitySerial() {
  float posBuf[NUM_SPHERES * 3];
  float velBuf[NUM_SPHERES * 3];
  float temp_velBuf[NUM_SPHERES * 3];
  float radBuf[NUM_SPHERES];
  struct Sphere s1, s2;
  unsigned i, j, k;
  float delta[3];
  float distanceSquared, radiusSumSquared;
  float s1Mass, s2Mass;
  float outVel[3];
  bool didImpact;
  struct timeval before, after;

  for(i = 0; i < 3; i++) {
    outVel[i] = 0;
  }

  // Get the buffers from the GPU
  cudaMemcpy(posBuf, d_posBuf, d_posBuf_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(velBuf, d_velBuf, d_velBuf_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(radBuf, d_radBuf, d_radBuf_size, cudaMemcpyDeviceToHost);

  gettimeofday(&before, NULL);

  for(i = 0; i < NUM_SPHERES; i++) {
    didImpact = false;

    outVel[0] = 0;
    outVel[1] = 0;
    outVel[2] = 0;

    s1.posX = posBuf[i * 3 + 0];
    s1.posY = posBuf[i * 3 + 1];
    s1.posZ = posBuf[i * 3 + 2];

    s1.velX = velBuf[i * 3 + 0];
    s1.velY = velBuf[i * 3 + 1];
    s1.velZ = velBuf[i * 3 + 2];

    s1.radius = radBuf[i];

    s1Mass = s1.radius * s1.radius * s1.radius;

    for(j = 0; j < NUM_SPHERES; j++) {
      if(j == i) {
        continue;
      }

      s2.posX = posBuf[j * 3 + 0];
      s2.posY = posBuf[j * 3 + 1];
      s2.posZ = posBuf[j * 3 + 2];
      
      s2.velX = velBuf[j * 3 + 0];
      s2.velY = velBuf[j * 3 + 1];
      s2.velZ = velBuf[j * 3 + 2];

      s2.radius = radBuf[j];

      delta[0] = s1.posX - s2.posX;
      delta[1] = s1.posY - s2.posY;
      delta[2] = s1.posZ - s2.posZ;

      distanceSquared = 0;
      radiusSumSquared = 0;

      for(k = 0; k < 3; k++) {
        distanceSquared = distanceSquared + delta[k] * delta[k];
      }

      radiusSumSquared = (s1.radius + s2.radius) * (s1.radius + s2.radius);

      if(distanceSquared < radiusSumSquared) {
        // We collide
        didImpact = true;

        s2Mass = s2.radius * s2.radius * s2.radius;

        outVel[0] = outVel[0] +
          (s1Mass - s2Mass) / (s1Mass + s2Mass) * s1.velX +
          (2 * s2Mass) / (s1Mass + s2Mass) * s2.velX;
        outVel[1] = outVel[1] +
          (s1Mass - s2Mass) / (s1Mass + s2Mass) * s1.velY +
          (2 * s2Mass) / (s1Mass + s2Mass) * s2.velY;
        outVel[2] = outVel[2] +
          (s1Mass - s2Mass) / (s1Mass + s2Mass) * s1.velZ +
          (2 * s2Mass) / (s1Mass + s2Mass) * s2.velZ;
      }
    }
    if(!didImpact) {
      for(k = 0; k < 3; k++) {
        outVel[k] = velBuf[i * 3 + k];
      }
    }

    for(k = 0; k < 3; k++) {
      temp_velBuf[i * 3 + k] = outVel[k];
    }

    didImpact = false;
  }

  gettimeofday(&after, NULL);

  printf("Total time elapsed: %.01f us\n", time_diff(before, after));

  // Copy temp_velBuf into GPU
  cudaMemcpy(d_velBuf, temp_velBuf, d_velBuf_size, cudaMemcpyHostToDevice);
}

__global__ void updatePositionKernel(float *d_posBuf, float *d_velBuf) {
  unsigned index, i;

  index = threadIdx.x + blockDim.x * blockIdx.x;

  // Each thread updates its portion
  if(index < NUM_SPHERES) {
    for(i = 0; i < 3; i++) {
      // d_posBuf[index * 3 + i] += d_velBuf[index * 3 + i];
      d_posBuf[index * 3 + i] += d_velBuf[index * 3 + i];;
    }
  }
}

void updatePosition() {
  unsigned numBlocks;

  numBlocks = (d_posBuf_size / 3) / BLOCK_SIZE;
  if((d_posBuf_size / 3) % BLOCK_SIZE) numBlocks++;

  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  // Kernel
  updatePositionKernel<<<dimGrid, dimBlock>>>(d_posBuf, d_velBuf);
}

void printPosition() {
  float temp_posBuf[d_posBuf_size];
  float temp_velBuf[d_velBuf_size];
  unsigned temp_idBuf[d_idBuf_size];
  unsigned i;

  // Copy from GPU to CPU
  cudaMemcpy(temp_posBuf, d_posBuf, d_posBuf_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_velBuf, d_velBuf, d_velBuf_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_idBuf, d_idBuf, d_idBuf_size, cudaMemcpyDeviceToHost);

  for(i = 0; i < NUM_SPHERES; i++) {
    printf("sphere %u\n", temp_idBuf[i]);
    printf("  pos: %f %f %f\n", temp_posBuf[i * 3 + 0], temp_posBuf[i * 3 + 1],
      temp_posBuf[i * 3 + 2]);
    printf("  vel: %f %f %f\n", temp_velBuf[i * 3 + 0], temp_velBuf[i * 3 + 1],
      temp_velBuf[i * 3 + 2]);
  }
}

void loop() {
  printf("--- TICK ---\n");
  updatePosition();
  // updateVelocity();
  updateVelocitySerial();
  // printPosition();
  // sleep(1);
}

int main() {
  // Initialize the object data
  struct Sphere spheres[NUM_SPHERES];
  unsigned i;  

  // Uncomment any of the test cases

  // Test case: Different velocities -----
  // UNCOMMENT START
  /* // First sphere
  spheres[0].posX = 0;
  spheres[0].posY = 0;
  spheres[0].posZ = 0;

  spheres[0].velX = 0.2;
  spheres[0].velY = 0;
  spheres[0].velZ = 0;

  spheres[0].radius = 1;

  spheres[0].id = 0;
  
  // Second sphere
  spheres[1].posX = 3;
  spheres[1].posY = 0;
  spheres[1].posZ = 0;
  
  spheres[1].velX = 0.1;
  spheres[1].velY = 0;
  spheres[1].velZ = 0;

  spheres[1].radius = 1;

  spheres[1].id = 1; */
  // UNCOMMENT END

  // Test case: Different directions -----
  // UNCOMMENT START
  /* // First sphere
  spheres[0].posX = 0;
  spheres[0].posY = 0;
  spheres[0].posZ = 0;

  spheres[0].velX = .1;
  spheres[0].velY = 0;
  spheres[0].velZ = 0;

  spheres[0].radius = 1;
 
  spheres[0].id = 0;

  // Second sphere
  spheres[1].posX = 2;
  spheres[1].posY = 4;
  spheres[1].posZ = 0;

  spheres[1].velX = 0;
  spheres[1].velY = -.1;
  spheres[1].velZ = 0;

  spheres[1].radius = 1;

  spheres[1].id = 1; */
  // UNCOMMENT END

  // Test case: Different masses -----
  // UNCOMMENT START
  // First sphere
  spheres[0].posX = 0;
  spheres[0].posY = 0;
  spheres[0].posZ = 0;
  
  spheres[0].velX = .1;
  spheres[0].velY = 0;
  spheres[0].velZ = 0;

  spheres[0].radius = 1;
  
  spheres[0].id = 0;

  // Second sphere
  spheres[1].posX = 3;
  spheres[1].posY = 0;
  spheres[1].posZ = 0;

  spheres[1].velX = 0;
  spheres[1].velY = 0;
  spheres[1].velZ = 0;

  spheres[1].radius = 1.26;

  spheres[1].id = 1;
  // UNCOMMENT END

  for(i = 2; i < 200; i++) {
    spheres[i].posX = i * 2;
    spheres[i].posY = i * 2;
    spheres[i].posZ = 10;

    spheres[i].velX = 1;
    spheres[i].velY = 1;
    spheres[i].velZ = 0;
    
    spheres[i].radius = 1;
    
    spheres[i].id = i;
  }

  // Create the position buffer
  float posBuf[NUM_SPHERES * 3];
  // Create the velocity buffer
  float velBuf[NUM_SPHERES * 3];
  // Create the radius buffer
  float radBuf[NUM_SPHERES];
  // Create the ID buffer
  unsigned idBuf[NUM_SPHERES];

  // Fill posradBuf
  for(i = 0; i < NUM_SPHERES; i++) {
    posBuf[i * 3 + 0] = spheres[i].posX;
    posBuf[i * 3 + 1] = spheres[i].posY;
    posBuf[i * 3 + 2] = spheres[i].posZ;
  }

  // Fill velocity buffer
  for(i = 0; i < NUM_SPHERES; i++) {
    velBuf[i * 3 + 0] = spheres[i].velX;
    velBuf[i * 3 + 1] = spheres[i].velY;
    velBuf[i * 3 + 2] = spheres[i].velZ;
  }

  // Fill radius buffer
  for(i = 0; i < NUM_SPHERES; i++) {
    radBuf[i] = spheres[i].radius;
  }

  // Fill id buffer
  for(i = 0; i < NUM_SPHERES; i++) {
    idBuf[i] = spheres[i].id;
  }

  // Allocate space on the GPU for storing these buffers
  d_posBuf_size = sizeof(float) * NUM_SPHERES * 3;
  d_velBuf_size = sizeof(float) * NUM_SPHERES * 3;
  d_radBuf_size = sizeof(float) * NUM_SPHERES;
  d_idBuf_size = sizeof(unsigned) * NUM_SPHERES;

  cudaMalloc((void **) &d_posBuf, d_posBuf_size);
  cudaMalloc((void **) &d_velBuf, d_velBuf_size);
  cudaMalloc((void **) &d_radBuf, d_radBuf_size);
  cudaMalloc((void **) &d_idBuf, d_idBuf_size);
  cudaMalloc((void **) &d_velBuf_temp, d_velBuf_size);

  // Copy into GPU
  cudaMemcpy(d_posBuf, posBuf, d_posBuf_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_velBuf, velBuf, d_velBuf_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_radBuf, radBuf, d_radBuf_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_idBuf, idBuf, d_idBuf_size, cudaMemcpyHostToDevice);

  /* while(true) {
    loop();
  } */

  for(i = 0; i < 100; i++) {
    loop();
  }

  cudaFree(d_posBuf);
  cudaFree(d_velBuf);
  cudaFree(d_radBuf);
  cudaFree(d_idBuf);
  cudaFree(d_velBuf_temp);

  return 0;
}

