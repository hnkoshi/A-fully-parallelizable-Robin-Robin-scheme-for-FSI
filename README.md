An Unconditionally Stable Explicit Robin–Robin Partitioned Scheme for Fluid–Structure Interaction

By Shihan Guo, Ping Lin, Yifan Wang, Xiaohe Yue and Haibiao Zheng

We propose an explicit partitioned (loosely coupled) scheme for fluid structure interaction (FSI) problems, specifically designed to achieve high computational efficiency in modern engineering simulations.
The FSI problem under consideration involves an incompressible viscous fluid, governed by the Navier-Stokes equations, with a thick linear elastic structure. 
The scheme adopts a Robin–Robin coupling condition, evaluating the right-hand side of the Robin boundary terms at each time step solely from the previous-step solutions. 
This explicit scheme allows the fluid and structure subproblems to be solved entirely independently within each time step, eliminating the need for staggered coupling or costly sub-iterations, which makes the method highly efficient and scalable for parallel computation. 
Various numerical experiments demonstrate the stability, accuracy, and superior computational efficiency of the proposed approach, highlighting its strong potential for large scale parallel FSI computations in engineering applications.
