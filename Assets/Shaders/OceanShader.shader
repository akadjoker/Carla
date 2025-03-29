Shader "Custom/OceanShader"
{
    Properties
    {
        _Color ("Color", Color) = (0.0, 0.5, 1.0, 0.8)
        _SpecColor ("Specular Color", Color) = (0.9, 0.9, 0.9, 1)
        _Shininess ("Shininess", Range(0.01, 1)) = 0.5
        _WaveSpeed ("Wave Speed", Range(0, 5)) = 1
        _WaveAmp ("Wave Amplitude", Range(0, 1)) = 0.2
        _WaveFreq ("Wave Frequency", Range(0, 10)) = 1
    }
    
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 200
        
        Blend SrcAlpha OneMinusSrcAlpha
        
        CGPROGRAM
        #pragma surface surf BlinnPhong alpha:fade vertex:vert
        #pragma target 3.0
        
        struct Input
        {
            float2 uv_MainTex;
            float3 worldPos;
        };
        
        fixed4 _Color;
        half _Shininess;
        float _WaveSpeed;
        float _WaveAmp;
        float _WaveFreq;
        
        void vert(inout appdata_full v) {
            float phase = _Time.y * _WaveSpeed;
            float offset = sin(v.vertex.x * _WaveFreq + phase) * _WaveAmp;
            offset += sin(v.vertex.z * _WaveFreq * 0.8 + phase * 1.2) * _WaveAmp;
            v.vertex.y += offset;
            
            // Calculate normals
            v.normal = normalize(float3(
                -cos(v.vertex.x * _WaveFreq + phase) * _WaveAmp * _WaveFreq,
                1,
                -cos(v.vertex.z * _WaveFreq * 0.8 + phase * 1.2) * _WaveAmp * _WaveFreq * 0.8
            ));
        }
        
        void surf(Input IN, inout SurfaceOutput o)
        {
            o.Albedo = _Color.rgb;
            o.Alpha = _Color.a;
            o.Specular = _Shininess;
            o.Gloss = 1.0;
        }
        ENDCG
    }
    FallBack "Diffuse"
}

