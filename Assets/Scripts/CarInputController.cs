using UnityEngine;
using UnityEngine.UI;

public class CarInputController : MonoBehaviour
{
    [Header("Referências")]
    private WheelCollider[] wheelColliders;
    private Rigidbody rb;
    
    [Header("Configurações de Controle")]
    public float motorForce = 1500f;          // Força do motor
    public float brakeForce = 3000f;          // Força de frenagem
    public float maxSteerAngle = 30f;         // Ângulo máximo de esterçamento
    public float steeringSpeed = 10f;         // Velocidade do esterçamento
    
    [Header("Configurações Avançadas")]
    public bool useAutomaticGearbox = true;    // Usar caixa automática
    public float[] gearRatios = { 3.5f, 2.5f, 1.8f, 1.4f, 1.1f, 0.85f };  // Relações de marcha
    public float reverseGearRatio = 3.5f;      // Relação de marcha ré
    public float downshiftRPM = 1500f;         // RPM para reduzir a marcha
    public float upshiftRPM = 4000f;           // RPM para aumentar a marcha
    public float maxRPM = 6000f;               // RPM máximo do motor
    public float idleRPM = 800f;               // RPM em marcha lenta
    
    [Header("Auxiliares de Direção")]
    public bool useAntiRollBars = true;        // Usar barras estabilizadoras
    public float antiRollForce = 5000f;        // Força das barras estabilizadoras
    public bool useTractionControl = true;     // Usar controle de tração
    public float tractionSlipLimit = 0.2f;     // Limite de escorregamento para controle de tração
    
    [Header("UI")]
    public Text speedText;                    // Texto da velocidade
    public Text gearText;                     // Texto da marcha
    public Text rpmText;                      // Texto do RPM
    public bool displayDebugInfo = true;      // Exibir informações de debug no console
    
    // Variáveis de estado
    private float currentSteerAngle = 0f;
    private float currentBrakeForce = 0f;
    private float currentMotorTorque = 0f;
    private int currentGear = 1;
    private float currentRPM = 0f;
    private float engineRPM = 0f;
    private bool isReversing = false;

 
    
    // Cache dos eixos de entrada
    public float steerInput = 0f;
    public float throttleInput = 0f;
    public float brakeInput = 0f;

    public float speed  = 0f;
    
    // Teclas de mudança de câmera
    private CameraFollow cameraFollow;
    

     public enum ControlMode
    {
        Web,
        Manual,
        Autopilot
    }
    public ControlMode currentControlMode = ControlMode.Manual;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError("Rigidbody não encontrado no objeto do carro.");
            return;
        }
        
        // Abaixar o centro de massa para estabilidade
        rb.centerOfMass = new Vector3(0, -0.5f, 0);
        
        // Encontrar todos os WheelColliders filhos
        wheelColliders = GetComponentsInChildren<WheelCollider>();
        if (wheelColliders.Length == 0)
        {
            Debug.LogError("Nenhum WheelCollider encontrado como filho do objeto do carro.");
        }
        
        // Procurar uma câmera que siga o carro
        cameraFollow = FindObjectOfType<CameraFollow>();
    }
    
    void Update()
    {
        // Capturar entradas

        
        if (IsManual())
        {
            steerInput = Input.GetAxis("Horizontal");
            throttleInput = Mathf.Clamp01(Input.GetAxis("Vertical"));  // Positivo para frente
            brakeInput = Mathf.Abs(Mathf.Clamp(Input.GetAxis("Vertical"), -1, 0));   
        }

            // Suavizar a direção
            float targetSteerAngle = steerInput * maxSteerAngle;
            currentSteerAngle = Mathf.Lerp(currentSteerAngle, targetSteerAngle, steeringSpeed * Time.deltaTime);
            
            // Verificar se está em marcha ré
            if (throttleInput < 0.1f && rb.velocity.magnitude < 1f && Input.GetKey(KeyCode.S))
            {
                isReversing = true;
            }
            else if (throttleInput > 0.1f && rb.velocity.magnitude < 1f)
            {
                isReversing = false;
            }
            
            // Câmbio automático
            if (useAutomaticGearbox)
            {
                AutomaticGearbox();
            }
            else
            {
             
                if (Input.GetKeyDown(KeyCode.E) && currentGear < gearRatios.Length)
                {
                    currentGear++;
                }
                else if (Input.GetKeyDown(KeyCode.Q) && currentGear > 1)
                {
                    currentGear--;
                }
                else if (Input.GetKeyDown(KeyCode.R))
                {
                    isReversing = !isReversing;
                }
            }
            
            // Atualizar RPM do motor baseado na velocidade da roda e relação de marcha
            UpdateEngineRPM();
        
        
        // Atualizar UI
        UpdateUI();
        
        // Trocar modos de câmera com teclas numéricas
        if (cameraFollow != null)
        {
            if (Input.GetKeyDown(KeyCode.Alpha1))
                cameraFollow.SetCameraMode(CameraFollow.CameraMode.Close);
            else if (Input.GetKeyDown(KeyCode.Alpha2))
                cameraFollow.SetCameraMode(CameraFollow.CameraMode.Far);
            else if (Input.GetKeyDown(KeyCode.Alpha3))
                cameraFollow.SetCameraMode(CameraFollow.CameraMode.Top);
            else if (Input.GetKeyDown(KeyCode.Alpha4))
                cameraFollow.SetCameraMode(CameraFollow.CameraMode.Side);
            else if (Input.GetKeyDown(KeyCode.Alpha5))
                cameraFollow.SetCameraMode(CameraFollow.CameraMode.First);
        }
    }
    
    void FixedUpdate()
    {
        // Calcular torque do motor conforme marcha
        float torqueMultiplier = isReversing ? -reverseGearRatio : gearRatios[currentGear - 1];
        
        // Limitar torque baseado no RPM (simular curva de torque)
        float rpmFactor = engineRPM / maxRPM;
        float torqueCurve = 1.0f;
        
        if (rpmFactor < 0.3f) 
            torqueCurve = rpmFactor / 0.3f; // Rampa de torque até 30% do RPM máximo
        else if (rpmFactor > 0.7f) 
            torqueCurve = 1.0f - ((rpmFactor - 0.7f) / 0.3f); // Redução de torque após 70% do RPM
            
        // Calcular torque final
        currentMotorTorque = throttleInput * motorForce * torqueMultiplier * torqueCurve;
        
        // Frenagem
        currentBrakeForce = brakeInput * brakeForce;
        
        // Aplicar valores às rodas
        ApplyToWheels();
        
        // Aplicar barras estabilizadoras se ativas
        if (useAntiRollBars && wheelColliders.Length >= 4)
        {
            ApplyAntiRoll();
        }
        
        // Controle de tração se ativo
        if (useTractionControl)
        {
            ApplyTractionControl();
        }
    }

    public bool IsAutopilot()
    {
        return currentControlMode == ControlMode.Autopilot;
    }

    public bool IsManual()
    {
        return currentControlMode == ControlMode.Manual;
    }

    public bool IsWeb()
    {
        return currentControlMode == ControlMode.Web;
    }

    public void SetAutopilot(bool enabled)
    {
        if (enabled)
            currentControlMode = ControlMode.Autopilot ;
    }

    public void SetManual()
    {
        currentControlMode = ControlMode.Manual;
     
    }

    public void SetWeb()
    {
        currentControlMode = ControlMode.Web;
 
    }
    
    private void ApplyToWheels()
    {
        if (wheelColliders.Length < 1) return;
      
        
        // Aplicar para cada roda
        for (int i = 0; i < wheelColliders.Length; i++)
        {
            WheelCollider wheel = wheelColliders[i];
            
            // Direção (assumindo que as duas primeiras rodas são dianteiras)
            if (i < 2)
            {
                wheel.steerAngle = currentSteerAngle;
            }
            
            // Torque do motor (assumindo que o veículo é traseiro, senão mudar para i < 2)
            if (i >= 2)
            {
                wheel.motorTorque = currentMotorTorque / 2; // Dividir entre as rodas motrizes
            }
            
            
            wheel.brakeTorque = currentBrakeForce;
            
            // Atualizar malhas das rodas (se tiver código para isso)
            // UpdateWheelMeshes();
        }
    }
    
    private void ApplyAntiRoll()
    {
        // Aplicar barras estabilizadoras para evitar rolagem excessiva nas curvas
        
        // Barra estabilizadora dianteira
        ApplyAntiRollBar(wheelColliders[0], wheelColliders[1]);
        
        // Barra estabilizadora traseira
        ApplyAntiRollBar(wheelColliders[2], wheelColliders[3]);
    }
    
    private void ApplyAntiRollBar(WheelCollider wheelL, WheelCollider wheelR)
    {
        // Obter as informações de compressão da suspensão
        WheelHit hitL, hitR;
        bool groundedL = wheelL.GetGroundHit(out hitL);
        bool groundedR = wheelR.GetGroundHit(out hitR);
        
        float travelL = groundedL ? (-wheelL.transform.InverseTransformPoint(hitL.point).y - wheelL.radius) / wheelL.suspensionDistance : 0f;
        float travelR = groundedR ? (-wheelR.transform.InverseTransformPoint(hitR.point).y - wheelR.radius) / wheelR.suspensionDistance : 0f;
        
        float antiRollForce = (travelL - travelR) * this.antiRollForce;
        
        // Aplicar forças opostas às rodas esquerda e direita
        if (groundedL)
            rb.AddForceAtPosition(wheelL.transform.up * -antiRollForce, wheelL.transform.position);
        if (groundedR)
            rb.AddForceAtPosition(wheelR.transform.up * antiRollForce, wheelR.transform.position);
    }
    
    private void ApplyTractionControl()
    {
        // Controle de tração para limitar deslizamento excessivo
        
        // Verificar as rodas traseiras (assumindo que são as motrizes)
        for (int i = 2; i < 4 && i < wheelColliders.Length; i++)
        {
            WheelHit wheelHit;
            wheelColliders[i].GetGroundHit(out wheelHit);
            
            // Calcular derrapagem (slip)
            float slip = Mathf.Abs(wheelHit.forwardSlip);
            
            // Se a derrapagem for maior que o limite, reduzir o torque
            if (slip > tractionSlipLimit)
            {
                // Reduzir o torque proporcionalmente ao excesso de derrapagem
                float reductionFactor = 1.0f - ((slip - tractionSlipLimit) / (1.0f - tractionSlipLimit));
                wheelColliders[i].motorTorque *= Mathf.Clamp01(reductionFactor);
            }
        }
    }
    
    private void AutomaticGearbox()
    {
        // Câmbio automático baseado no RPM
        
        if (!isReversing)
        {
         
            if (engineRPM < idleRPM && rb.velocity.magnitude < 1f && throttleInput > 0.1f)
            {
                currentGear = 1;
            }
     
            else if (engineRPM > upshiftRPM && currentGear < gearRatios.Length)
            {
                currentGear++;
                engineRPM = Mathf.Lerp(downshiftRPM, upshiftRPM, 0.4f); // Simular queda de RPM na troca
            }
      
            else if (engineRPM < downshiftRPM && currentGear > 1)
            {
                currentGear--;
                engineRPM = Mathf.Lerp(downshiftRPM, upshiftRPM, 0.6f); // Simular aumento de RPM na troca
            }
        }
        else
        {
            // Primeira para marcha a ré
            if (rb.velocity.magnitude < 1f && throttleInput < 0.1f && brakeInput > 0.1f)
            {
                isReversing = true;
            }
        }
    }
    
    private void UpdateEngineRPM()
    {
        if (wheelColliders.Length < 3) return;
        
        // Obter RPM médio das rodas motrizes
        float wheelRPM = (wheelColliders[2].rpm + wheelColliders[3].rpm) / 2f;
        
        // Se está parado e sem aceleração, usar idle RPM
        if (Mathf.Abs(wheelRPM) < 0.1f && throttleInput < 0.1f)
        {
            currentRPM = idleRPM;
        }
        else
        {
            // Calcular RPM baseado na velocidade da roda e relação de marcha
            float ratio = isReversing ? reverseGearRatio : gearRatios[currentGear - 1];
            currentRPM = Mathf.Abs(wheelRPM * ratio * 3.6f); // Fator 3.6 para ajustar a escala
            
            // Limitar o RPM
            currentRPM = Mathf.Clamp(currentRPM, idleRPM, maxRPM);
        }
        
        // Suavizar a transição do RPM
        engineRPM = Mathf.Lerp(engineRPM, currentRPM, Time.deltaTime * 5f);
        
        // Adicionar efeito de RPM ao acelerar parado
        if (Mathf.Abs(wheelRPM) < 5f && throttleInput > 0.1f)
        {
            engineRPM += throttleInput * Time.deltaTime * 3000f;
            engineRPM = Mathf.Clamp(engineRPM, idleRPM, maxRPM);
        }
    }
    
    private void UpdateUI()
    {
        if (!displayDebugInfo) return;
        
        // Velocidade em km/h
        speed= rb.velocity.magnitude * 3.6f; // Converter de m/s para km/h
        
        if (speedText != null)
        {
            speedText.text = $"{speed:0} km/h";
        }
        
        if (gearText != null)
        {
            gearText.text = isReversing ? "R" : currentGear.ToString();
        }
        
        if (rpmText != null)
        {
            rpmText.text = $"{engineRPM:0} RPM";
        }
        
        // Debug info no console
       
      
    }
    
    // Função para obter a velocidade atual em km/h
    public float GetCurrentSpeed()
    {
        return rb.velocity.magnitude * 3.6f;
    }
    
    // Função para obter o RPM atual
    public float GetCurrentRPM()
    {
        return engineRPM;
    }
    
    // Função para obter a marcha atual
    public string GetCurrentGear()
    {
        return isReversing ? "R" : currentGear.ToString();
    }
    
    public void SetAutopilotControls(float throttle, float brake, float steer)
    {
         if (IsManual()) return;
         Debug.Log($" Throttle: {throttle:0.00} | Brake: {brake:0.00} | Steer: {steer:0.00}");

        steerInput = steer;
        throttleInput = throttle;
        brakeInput = brake;
    }
}