#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <Windows.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <random>
#include <ctime>
#include <cstdlib>
using namespace std;

#define W 120
#define H 40

#define N 100
#define xmin -101.0
#define xmax 114.0
#define ymin -137.0
#define ymax 139.0
#define t0 0.0
#define tmax 1000.0
#define tau 0.5
#define m 1.0
#define A1 23.0
#define A2 22.0
#define p1 5.0
#define p2 7.0

void Draw(double* U) {
    static char screen[H][W + 1];

    // 1. Очистить экран
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++)
            screen[y][x] = '.';
        screen[y][W] = '\0';
    }

    // 2. Нарисовать объекты
    for (int i = 0; i < N; i++) {
        double x = U[i * 4 + 2];
        double y = U[i * 4 + 3];

        // Масштабирование в диапазон
        int sx = (int)((x - xmin) * (W - 1) / (xmax - xmin));
        int sy = (int)((y - ymin) * (H - 1) / (ymax - ymin));

        //(0,0) - верхний левый угол в консоли
        sy = H - 1 - sy; // инвертация, чтоб движения вверх не казались движениями вниз

        // Проверка на границы
        if (sx >= 0 && sx < W && sy >= 0 && sy < H)
            screen[sy][sx] = '*';
    }

    // 3. Вывод карты
    for (int y = 0; y < H; y++)
        std::cout << screen[y] << "\n";
}


__host__ double GetRandomDouble(double min, double max)
{
    double p = min + (max - min) * ((double)rand() / RAND_MAX);
    return p;
}

// CUDA-ядро (выполняется на GPU)
__global__ void Kernel(double* U, double* Unew)
{
    // Вычисляем индекс текущей частицы в массиве
    // threadIdx.x - индекс потока в блоке
    // blockIdx.x - индекс блока в сетке  
    // blockDim.x - размер блока
    int k = 4;
    int i = k * (threadIdx.x + blockIdx.x * blockDim.x);
    double Vx = 0.0, Vy = 0.0;
    __syncthreads();    // Синхронизация всех потоков в блоке

    // Проверяем, что индекс в пределах массива - блок может простаивать.
    if (i < N * k)
    {
        // Расчет сил - КАЖДАЯ ЧАСТИЦА ВЗАИМОДЕЙСТВУЕТ СО ВСЕМИ ОСТАЛЬНЫМИ
        for (int j = 0; j < N * k; j += k)
        {
            if (j != i)
            {
                double dx = U[j + 2] - U[i + 2];
                double dy = U[j + 3] - U[i + 3];
                double dist = sqrt(dx * dx + dy * dy);
                if (dist < 1e-15) continue;

                // Та же физическая модель сил
                Vx += ((A1 / m) * dx / pow(dist, p1)) - ((A2 / m) * dx / pow(dist, p2));
                Vy += ((A1 / m) * dy / pow(dist, p1)) - ((A2 / m) * dy / pow(dist, p2));
            }
        }

        // Обновление скоростей и координат (метод Эйлера)
        Unew[i] = U[i] + tau * Vx;
        Unew[i + 1] = U[i + 1] + tau * Vy;
        Unew[i + 2] = U[i + 2] + tau * Unew[i];
        Unew[i + 3] = U[i + 3] + tau * Unew[i + 1];

        // Обработка столкновений со стенками (та же логика)
        if (Unew[i + 2] > xmax)
        {
            Unew[i] = -Unew[i];
            double delta = Unew[i + 2] - xmax;
            Unew[i + 2] = xmax - delta;
        }
        if (Unew[i + 2] < xmin)
        {
            Unew[i] = -Unew[i];
            double delta = Unew[i + 2] - xmin;
            Unew[i + 2] = xmin - delta;
        }
        if (Unew[i + 3] > ymax)
        {
            Unew[i + 1] = -Unew[i + 1];
            double delta = Unew[i + 3] - ymax;
            Unew[i + 3] = ymax - delta;
        }
        if (Unew[i + 3] < ymin)
        {
            Unew[i + 1] = -Unew[i + 1];
            double delta = Unew[i + 3] - ymin;
            Unew[i + 3] = ymin - delta;
        }
    }
    __syncthreads();    // Синхронизация перед копированием
    if (i < N * k)  // Копируем новые значения в основной массив
    {
        for (int a = 0; a < k; a++)
            U[i + a] = Unew[i + a];
    }
    __syncthreads();    // Финальная синхронизация
}

int main()
{
    SetConsoleOutputCP(1251);
    srand(time(0));
    int r = 4;
    int size = N * r * sizeof(double);  // Размер в байтах

    // Выделяем память на CPU (хосте)
    double* U = new double[N * r];
    for (int i = 0; i < N * r; i++)
        U[i] = 0.0; //Vx Vy ... = 0

    // Инициализация случайных позиций частиц
    int l = 0;
    while (l < N)
    {
        U[r * l + 2] = GetRandomDouble(xmin, xmax); //х
        U[r * l + 3] = GetRandomDouble(ymin, ymax); //y
        l++;
    }

    // ВЫДЕЛЕНИЕ ПАМЯТИ НА GPU
    double* UDev = nullptr, * UnewDev = nullptr;
    
    cudaMalloc((void**)&UDev, size);    // Память для текущего состояния
    cudaMalloc((void**)&UnewDev, size); // Память для нового состояния
    float timet;
    cudaEvent_t tn, tk;
    cudaEventCreate(&tn);// Создаем события для точного замера времени
    cudaEventCreate(&tk);

    // Копируем начальные данные с CPU на GPU
    cudaMemcpy(UDev, U, size, cudaMemcpyHostToDevice);

    cudaEventRecord(tn, 0); // Запускаем таймер

    // ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    int step_count = 0; //т.к. С++ не умеет брать % 2 у double
    int block_size = 32;    // 32 потока в блоке
    for (double t = t0; t <= tmax; t += tau, step_count++)
    {
        // ЗАПУСК CUDA-ЯДРА
        // <<<число_блоков, число_потоков_в_блоке>>>
        Kernel << < (int)(N / block_size) + 1, block_size >> > (UDev, UnewDev);

        // Синхронизация - ждем завершения ядра
        cudaDeviceSynchronize();
        

        //чтоб отобразить в консоли - получим обратно с устройства
        cudaMemcpy(U, UDev, size, cudaMemcpyDeviceToHost);
                // очистка консоли
        #ifdef _WIN32
                system("cls");
        #else
                system("clear");
        #endif
        if (step_count % 2 == 0) Draw(U);   // Рисуем каждый второй кадр
                cout << "t = " << t << endl;
    }

    // ОСТАНОВКА ТАЙМЕРА
    cudaEventRecord(tk, 0);
    cudaEventSynchronize(tk);
    cudaEventElapsedTime(&timet, tn, tk);   // Время в миллисекундах
    cudaEventDestroy(tn);
    cudaEventDestroy(tk);

    // Копируем финальные результаты с GPU на CPU
    cudaMemcpy(U, UDev, size, cudaMemcpyDeviceToHost);

    // Освобождаем память на GPU
    cudaFree(UDev);
    cudaFree(UnewDev);

    // Сохранение результатов в файл
    char filename[256];
    sprintf_s(filename, "Result.txt");
    ofstream FileResult(filename);
    if (!FileResult.is_open())
    {
        cout << "Error opening the file for writing!\n";
        exit(1);
    }
    FileResult << "Time = " << timet / 1000.0 << " s" << endl;
    l = 1;
    for (int i = 2; i < N * r; i += r, l++)
    {
        FileResult << "Object number " << l << ": (" << U[i] << ", " << U[i + 1] << ")" << endl;
    }
    FileResult.close();

    delete[] U;
    return 0;
}
