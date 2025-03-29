using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    [Header("Alvo")]
    public Transform target;
    
    [Header("Posicionamento")]
    public Vector3 offset = new Vector3(0, 2, -5); // Posição relativa ao alvo
    public float heightDamping = 2.0f;            // Suavização da altura
    public float rotationDamping = 3.0f;          // Suavização da rotação
    
    [Header("Opções de Visualização")]
    public float distance = 5.0f;                 // Distância do alvo
    public float height = 2.0f;                   // Altura sobre o alvo
    public float lookAheadDistance = 3.0f;        // Quanto à frente olhar
    
    [Header("Comportamento")]
    public bool lookAtTarget = true;              // Se deve olhar para o alvo
    public bool smoothFollow = true;              // Se deve seguir suavemente
    public float smoothSpeed = 10.0f;             // Velocidade de suavização

    [Header("Configurações de Cinematic")]
    public bool enableCinematicMode = false;      // Ativar modos cinematográficos
    public float fovChangeSpeed = 2.0f;           // Velocidade de mudança do FOV
    public float baseFOV = 60f;                   // FOV padrão
    public float speedFOVFactor = 0.05f;          // Quanto a velocidade afeta o FOV
    
    private Camera cam;
    private Vector3 currentVelocity;
    private float initialFOV;
    
    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("Câmera não encontrada no objeto. Adicione um componente Camera.");
        }
        
        if (target == null)
        {
            Debug.LogWarning("Alvo não definido para a câmera de seguimento.");
        }
        
        initialFOV = cam.fieldOfView;
    }
    
    void LateUpdate()
    {
        if (target == null) return;
        
        // Calcular ângulo desejado
        float wantedRotationAngle = target.eulerAngles.y;
        float wantedHeight = target.position.y + height;
        
        float currentRotationAngle = transform.eulerAngles.y;
        float currentHeight = transform.position.y;
        
        // Suavizar rotação
        if (smoothFollow)
        {
            currentRotationAngle = Mathf.LerpAngle(currentRotationAngle, wantedRotationAngle, 
                                                  rotationDamping * Time.deltaTime);
            
            // Suavizar altura
            currentHeight = Mathf.Lerp(currentHeight, wantedHeight, 
                                      heightDamping * Time.deltaTime);
        }
        else
        {
            currentRotationAngle = wantedRotationAngle;
            currentHeight = wantedHeight;
        }
        
        // Converter o ângulo em rotação
        Quaternion currentRotation = Quaternion.Euler(0, currentRotationAngle, 0);
        
        // Calcular posição da câmera baseada na distância
        Vector3 targetPos = target.position;
        
        // Adicionar look ahead (olhar à frente baseado na direção do alvo)
        if (lookAheadDistance > 0)
        {
            targetPos += target.forward * lookAheadDistance;
        }
        
        // Definir posição baseada na distância, altura e rotação
        Vector3 wantedPosition = targetPos - (currentRotation * Vector3.forward * distance);
        wantedPosition.y = currentHeight;
        
        // Se usando offset personalizado em vez dos cálculos acima
        if (offset != Vector3.zero && !smoothFollow)
        {
            wantedPosition = target.position + offset;
        }
        
        // Mover a câmera suavemente para a posição
        if (smoothFollow)
        {
            transform.position = Vector3.SmoothDamp(transform.position, wantedPosition, 
                                                   ref currentVelocity, smoothSpeed * Time.deltaTime);
        }
        else
        {
            transform.position = wantedPosition;
        }
        
        // Fazer a câmera olhar para o alvo
        if (lookAtTarget)
        {
            transform.LookAt(targetPos);
        }
        
        // Efeitos de FOV baseados na velocidade (modo cinematográfico)
        if (enableCinematicMode && cam != null)
        {
            Rigidbody targetRb = target.GetComponent<Rigidbody>();
            if (targetRb != null)
            {
                float speedFactor = targetRb.velocity.magnitude * speedFOVFactor;
                float targetFOV = Mathf.Clamp(baseFOV + speedFactor, baseFOV, baseFOV + 30);
                cam.fieldOfView = Mathf.Lerp(cam.fieldOfView, targetFOV, fovChangeSpeed * Time.deltaTime);
            }
        }
    }
    
    // Função para alternar rapidamente entre diferentes modos de câmera
    public void SetCameraMode(CameraMode mode)
    {
        switch (mode)
        {
            case CameraMode.Close:
                distance = 3.0f;
                height = 1.5f;
                lookAheadDistance = 2.0f;
                break;
                
            case CameraMode.Far:
                distance = 8.0f;
                height = 3.0f;
                lookAheadDistance = 4.0f;
                break;
                
            case CameraMode.Top:
                distance = 2.0f;
                height = 10.0f;
                lookAheadDistance = 0.0f;
                break;
                
            case CameraMode.Side:
                distance = 4.0f;
                height = 1.0f;
                offset = new Vector3(5, 1, 0);
                lookAheadDistance = 0.0f;
                break;
                
            case CameraMode.First:
                distance = 0.1f;
                height = 0.75f;
                lookAheadDistance = 0.0f;
                offset = new Vector3(0, 0.75f, 0.1f);
                break;
        }
    }
    
    // Configurações predefinidas da câmera
    public enum CameraMode
    {
        Close,  // Perto do carro
        Far,    // Longe do carro
        Top,    // Vista superior
        Side,   // Vista lateral
        First   // Primeira pessoa
    }
}