using UnityEngine;

public class BasicLensFlare : MonoBehaviour
{
    public float brightness = 1.5f;      // Brilho do flare
    public float fadeSpeed = 2.0f;       // Velocidade de fade do flare
    public Color tint = Color.white;     // Tint do flare (cor)
    
    private LensFlare lensFlare;
    
    void Start()
    {
        // Obtém ou adiciona o componente LensFlare
        lensFlare = GetComponent<LensFlare>();
        if (lensFlare == null)
            lensFlare = gameObject.AddComponent<LensFlare>();
            
        // Configura os parâmetros
        lensFlare.brightness = brightness;
        lensFlare.fadeSpeed = fadeSpeed;
        lensFlare.color = tint;
    }
}