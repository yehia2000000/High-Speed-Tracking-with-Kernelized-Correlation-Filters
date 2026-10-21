#pragma once

#include "riscv_port.hpp"
#include "types.hpp"
#include <cstddef>
#include <cstdint>

enum class ImageType
{
    GRAY,
    RGB,
};

template <typename T, ImageType Type = ImageType::GRAY>
class Image
{
private:
    int width;
    int height;
    int stride;

    T* data;

    static constexpr int Channels()
    {
        return Type == ImageType::GRAY ? 1 : 3;
    }

    int index(const int x, const int y, const int c = 0) const
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        assert(c >= 0 && c < Channels());
        return y * stride + x * Channels() + c;
    }

    void Allocate(int _width, int _height)
    {
        assert(_width > 0 && _height > 0);
        width = _width;
        height = _height;
        stride = _width * Channels();
        data = new T[stride * _height];
    }

    static int MapCoord(int p , const int len , const BorderType type){
        if (p >= 0 && p<len) return p ; 

        switch(type){
            case BorderType::NO_BORDER : 
            return -1 ; 

            case BorderType::REPLICATE : 
            return p<0 ? 0 :len -1 ; 

            case BorderType::CONSTANT : 
            return -1 ; 

            case BorderType::MIRROR:
            if (len == 1) return 0;
            while (p < 0 || p >= len)   
                p = (p < 0) ? -p : (2 * len - 2 - p);
            return p; 

        }
        return -1 ; 

    }

    


public:
    Image() : width(0), height(0), stride(0), data(nullptr) {}
    Image(int width, int height)
    {
        Allocate(width, height);
    }

    ~Image()
    {
        delete[] data;
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& other) noexcept
        : width(other.width), height(other.height),
          stride(other.stride), data(other.data)
    {
        other.width = 0;
        other.height = 0;
        other.stride = 0;
        other.data = nullptr;
    }

    Image& operator=(Image&& other) noexcept
    {
        if (this != &other)
        {
            delete[] data;
            width = other.width;
            height = other.height;
            stride = other.stride;
            data = other.data;
            other.width = 0;
            other.height = 0;
            other.stride = 0;
            other.data = nullptr;
        }
        return *this;
    }

    T GetPixelBorder (const int x , const int  y ,  const int c =0 , 
                      const BorderType type  =  BorderType::NO_BORDER ,const T & value =T{})const
    {
      const int sx = MapCoord(x,width,type); 
      const int sy = MapCoord(y, height, type); 
      if(sx<0 || sy<0) return value ; 
      return data[index(sx,sy,c)];
    }

    void MakeBorder (Image &out ,int pad ,  const BorderType type = BorderType::NO_BORDER ,  const T& value = T{})const {
        if (type == BorderType::NO_BORDER) pad = 0; 
        for (int y =0 ; y<out.height ; ++y)
            for (int x = 0; x < out.width; ++x)
                for (int c = 0; c < Channels(); ++c)
                    out.SetPixel(x, y,
                        GetPixelBorder(x - pad, y - pad, c, type, value), c);
    }
              
    int GetWidth() {
        return width ; 
    }
    int GetHeight(){
        return height ; 
    }
    int Getsize() const
    {
        return width * height;
    }
    T GetPixel_index(const int index) const
    {
        return data[index];
    }

    T GetPixel(const int x, const int y, const int c = 0) const
    {
        return data[index(x, y, c)];
    }

    void SetPixel(const int x, const int y, const T& value, const int c = 0)
    {
        data[index(x, y, c)] = value;
    }
    void SetPixel_index(const int index, const T& value)
    {
        data[index] = value;
    }

    T* GetPtr(const int x, const int y)
    {
        return data + index(x, y);
    }

    const T* GetPtr(const int x, const int y) const
    {
        return data + index(x, y);
    }

#ifdef RISCV_QEMU
    // PGM/PPM file I/O. Only available on qemu-user (hosted libc).
    // Read accepts ASCII (P2/P3) and binary (P5/P6). Write always binary.
    void Read(const char* path);
    void Write(const char* path) const;
#else
    void Read(int new_width, int new_height, const uint8_t* buffer);
#endif

    int Width() const noexcept { return width; }
    int Height() const noexcept { return height; }
};