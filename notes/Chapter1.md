## Chapter 1: Introduction
### Part 1.0: Motivation
Not too much learning here, just restating the well known background that CPUs used to be sequencial but as we scaled up our compute we needed more and more, and parallel computing started revolution to unlock compute

### Part 1.1: Hetrogeneous Parallel Computing
The section describes the two paths to parallel computing:
1. Latency Driven optimization (traditional multi-core CPUs)
2. Throughput driven optimization (GPUs)

The section acknowledges that CPUs and GPUs thrive in different settings, where CPU optimizes latency and reserves chip space for additional caches (L2, L3, etc), implements out-of-order execution with branch prediction and other techniques, whereas GPUs maximize throughput and allow really high FLOPS thus thriving in numerically heavy bits of programs.

The section introduces CUDA as a software interface that's revolutionary in that it replaced the need to use a graphics API to interface with a GPU and allowed general purpose non-graphics GPU programms to be written using NVIDIA chips.

### Part 1.2: Why more speed
The answer is simple: Our problems are getting more complex and iterating on problems is starting to require more compute. Some problems: (Deep Learning, Graphics, MRIs, etc) are natrually parallel problems that parallel computation can make a significant improvment in.

### Part 1.3: Speeding up real applications
This section explains Amdahls law and that CPUs and GPUs work together. Specifically that your optimization using parallelization is bottleknecked by the sequential part of the application.


### The rest of the chapter
The rest of the chapter describes:
- Challenges in parallel computing, especially when programs are memory bound and when the parallel program executes more than its sequential coutnerpart
- Topics in the book and how the book will teach them.
