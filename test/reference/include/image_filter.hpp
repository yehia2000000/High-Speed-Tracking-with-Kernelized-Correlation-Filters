    #ifndef  IMAGE_FILTER_H_ 
    #define IMAGE_FILTER_H_

    #include "image.hpp"  
    #include <cstdint>

    template <int K> 
    struct Kernel {
        int32_t w[K * K]; 
    };

    namespace ref
    {

        template <int K >
        bool  Image_filter (Image <uint8_t,ImageType::GRAY> & Input , Image <uint8_t, ImageType::GRAY> & Output , 
                            const Kernel<K> ker,BorderType border_mode , const uint8_t value = 0  )
        {

        bool LocalErrorState = true ; 
        

        if  (Input.GetPtr(0,0) != nullptr){
            int size_image  =  Input.Getsize() ; 
            int size_k = K ;
            int radius =  size_k /2 ; 
            int kArea = size_k *size_k ; 
            
            const int pad = (border_mode == BorderType::NO_BORDER) ? 0 : radius;

            const Image<uint8_t, ImageType::GRAY> padded = Input.MakeBorder(pad, border_mode, value);
            
            const int outW = padded.Width()  - 2 * radius;
            const int outH = padded.Height() - 2 * radius;
            Output = Image<uint8_t, ImageType::GRAY>(outW, outH);

            for (int32_t y = 0; y < outH; y++)
            {
                for (int32_t x = 0; x < outW; x++)
                {
                    int32_t  sum = 0;
                    for (int32_t ky = -radius; ky <= radius; ky++)
                    {
                        for (int32_t kx = -radius; kx <= radius; kx++)
                        {
                            sum += padded.GetPixel(x+ radius-kx,y+ radius-ky) * ker.w[(ky+radius)*K +(kx+radius)];
                        }
                    }
                    sum /= kArea;
                    if (sum < 0)   sum = 0;
                    if (sum > 255) sum = 255;
                    Output.SetPixel(x, y, static_cast<uint8_t>(sum));
                }
            }


        }
        else {
        LocalErrorState = false ; 
        }

        return LocalErrorState ; 


        }
        
    }


    #endif