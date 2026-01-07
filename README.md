# CME 192: MATLAB for Scientific Computing and Engineering (Winter 2026)

Welcome to **CME 192** — an 8-lecture, application-driven short course designed to make you **fluent in MATLAB** and comfortable building **real scientific computing workflows** (not just toy exercises).

This repository is the central home for:
- **Lecture slides**
- **MATLAB Live Scripts (`.mlx`)**
- **Required datasets**
- **Assignment handouts + starter code**
- Any other course content needed to succeed in the class

---

## Course info (quick reference)

- **Instructor:** John Winnicki — winnicki@stanford.edu  
- **Quarter:** Winter 2026  
- **Meetings:** Thursdays 4:30–5:20 PM  
- **Location:** Hewlett Teaching Center 102  
- **Primary course hub:** Course website (Canvas updated periodically)

> **Support:** Use **Ed Discussion** for Q&A (monitored daily). Email the instructor for confidential issues.

---

## What you’ll learn

This course emphasizes **real-world applications** of applied & computational mathematics using **MATLAB’s ecosystem**.

Topics include:
1. Advanced plotting and 2D/3D visualization + interactive plotting  
2. Numerical linear algebra, ODEs/PDEs, symbolic math  
3. Big data and databases  
4. Python/C++ interfaces and workflows  
5. Statistics and machine learning  
6. Optimization and simulation/modeling  
7. Image processing and signal processing  
8. Parallel processing (multicore + GPU)

By the end of the course, you should be able to build complete workflows in MATLAB:
**load → clean → visualize → analyze/model → interpret → present results**.

---

## Repository structure

```
.
├── lectures/
│   ├── lecture01_intro/
│   │   ├── slides.pdf
│   │   ├── live_script.mlx
│   │   └── data/
│   ├── lecture02_plotting/
│   ├── lecture03_nla_odes_symbolic/
│   ├── lecture04_bigdata_interfaces_ml/
│   ├── lecture05_stats_ml/
│   ├── lecture06_optimization_simulation/
│   ├── lecture07_image_signal/
│   └── lecture08_parallel_interactive/
├── assignments/
│   ├── assignment1/
│   │   ├── README.md
│   │   ├── starter_code/
│   │   └── data/
│   └── assignment2/
│       ├── README.md
│       ├── starter_code/
│       └── data/
├── data/
│   ├── shared/        # datasets used across multiple lectures/assignments
│   └── external/      # large datasets (see note below)
├── resources/
│   ├── matlab_setup.md
│   ├── matlab_onramp.md
│   └── matlab_python_cheatsheet.md
├── policies/
│   ├── syllabus.md
│   ├── grading.md
│   ├── late_policy.md
│   └── integrity.md
└── README.md
```

---

## Lecture schedule

| Lecture | Topic |
|--------:|------|
| 1 | Course introduction; MATLAB setup/capabilities; why should you care? |
| 2 | Advanced plotting and visualizations in MATLAB |
| 3 | Numerical linear algebra, ODEs/PDEs, and symbolic math |
| 4 | Big data, Python/C++ in MATLAB, intro to machine learning |
| 5 | Statistics and machine learning in MATLAB |
| 6 | Optimization and simulation/modeling |
| 7 | Image processing and signal processing |
| 8 | Parallel processing (multicore & GPU), interactive plotting |

Lecture materials are posted under `lectures/lectureXX_*/` as:
- `slides.pdf`  
- `live_script.mlx`  
- any lecture-specific data in `data/`

---

## License

Course materials are intended for enrolled students and course staff. Redistribution is not permitted unless explicitly stated.
