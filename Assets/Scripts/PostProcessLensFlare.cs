using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PostProcessLensFlare : MonoBehaviour
{
    public Shader lensFlareShader;          // Shader para o efeito de lens flare
    public Color flareColor = new Color(1.0f, 0.8f, 0.6f, 1.0f);  // Cor do lens flare
    public float flareBrightness = 1.0f;    // Brilho do efeito
    public float ghostCount = 3.0f;         // Número de "fantasmas" no lens flare
    public float ghostDispersal = 0.6f;     // Dispersão dos fantasmas
    public float haloWidth = 0.4f;          // Largura do halo
    
    private Material lensFlaresMaterial;
    private Light sunLight;
    
    void Start()
    {
        // Criar material com o shader
        if (lensFlareShader != null)
            lensFlaresMaterial = new Material(lensFlareShader);
        else
            Debug.LogError("Shader de Lens Flare não atribuído!");
            
        // Procurar uma luz direcional para usar como sol
        Light[] lights = FindObjectsOfType<Light>();
        foreach (Light light in lights)
        {
            if (light.type == LightType.Directional)
            {
                sunLight = light;
                break;
            }
        }
    }
    
    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (lensFlaresMaterial != null && sunLight != null)
        {
            // Calcular a posição do sol na tela
            Vector3 sunScreenPos = Camera.main.WorldToViewportPoint(sunLight.transform.forward * -100000 + transform.position);
            
            // Configurar os parâmetros do shader
            lensFlaresMaterial.SetColor("_FlareColor", flareColor);
            lensFlaresMaterial.SetFloat("_Brightness", flareBrightness);
            lensFlaresMaterial.SetFloat("_GhostCount", ghostCount);
            lensFlaresMaterial.SetFloat("_GhostDispersal", ghostDispersal);
            lensFlaresMaterial.SetFloat("_HaloWidth", haloWidth);
            lensFlaresMaterial.SetVector("_SunPosition", new Vector4(sunScreenPos.x, sunScreenPos.y, 0, 0));
            
            // Aplicar o efeito
            Graphics.Blit(source, destination, lensFlaresMaterial);
        }
        else
        {
            // Se o material não estiver pronto, apenas copiar a textura de origem para o destino
            Graphics.Blit(source, destination);
        }
    }
}