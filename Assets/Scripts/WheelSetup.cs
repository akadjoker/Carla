using UnityEngine;

public class WheelSetup : MonoBehaviour
{
    [System.Serializable]
    public class WheelData
    {
        public WheelCollider collider;
        public Transform wheelMesh;
        public bool isFrontWheel = false;
    }

    [Header("Rodas")]
    public WheelData[] wheels;

    [Header("Configurações de Suspensão")]
    public float suspensionDistance = 0.3f;
    public float suspensionSpring = 35000f;
    public float suspensionDamper = 4500f;
    public float suspensionTargetPosition = 0.5f;

    [Header("Configurações de Fricção")]
    public float forwardFriction = 1.5f;
    public float sidewaysFriction = 1.5f;
    public float forwardExtremumSlip = 0.4f;
    public float sidewaysExtremumSlip = 0.4f;
    public float forwardAsymptoteSlip = 0.8f;
    public float sidewaysAsymptoteSlip = 0.8f;
    public float forwardStiffness = 1.0f;
    public float sidewaysStiffness = 1.0f;

    [Header("Configurações Avançadas")]
    public float wheelMass = 20f;
    public float wheelRadius = 0.4f;
    public float wheelDampingRate = 0.25f;
    
    private Rigidbody rb;

    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError("Não foi encontrado um Rigidbody no objeto do carro. Adiciona um Rigidbody ao objeto pai.");
            return;
        }

        // Configurar o centro de massa mais baixo para melhor estabilidade
        rb.centerOfMass = new Vector3(0, -0.5f, 0);
        
        ConfigureWheels();
    }

    public void ConfigureWheels()
    {
        foreach (WheelData wheel in wheels)
        {
            if (wheel.collider == null) continue;
            
            // Configurar a suspensão
            JointSpring spring = wheel.collider.suspensionSpring;
            spring.spring = suspensionSpring;
            spring.damper = suspensionDamper;
            spring.targetPosition = suspensionTargetPosition;
            wheel.collider.suspensionSpring = spring;
            wheel.collider.suspensionDistance = suspensionDistance;
            
            // Configurar propriedades físicas básicas
            wheel.collider.mass = wheelMass;
            wheel.collider.radius = wheelRadius;
            wheel.collider.wheelDampingRate = wheelDampingRate;
            
            // Configurar fricção para frente
            WheelFrictionCurve forwardCurve = wheel.collider.forwardFriction;
            ConfigureFrictionCurve(ref forwardCurve, forwardFriction, forwardExtremumSlip, 
                                  forwardAsymptoteSlip, forwardStiffness);
            wheel.collider.forwardFriction = forwardCurve;
            
            // Configurar fricção lateral
            WheelFrictionCurve sidewaysCurve = wheel.collider.sidewaysFriction;
            ConfigureFrictionCurve(ref sidewaysCurve, sidewaysFriction, sidewaysExtremumSlip, 
                                  sidewaysAsymptoteSlip, sidewaysStiffness);
            wheel.collider.sidewaysFriction = sidewaysCurve;
        }
    }
    
    private void ConfigureFrictionCurve(ref WheelFrictionCurve curve, float friction, 
                                      float extremumSlip, float asymptoteSlip, float stiffness)
    {
        curve.extremumSlip = extremumSlip;
        curve.extremumValue = friction;
        curve.asymptoteSlip = asymptoteSlip;
        curve.asymptoteValue = friction * 0.8f; // Geralmente um pouco menor que extremum
        curve.stiffness = stiffness;
    }
    
    private void Update()
    {
        // Atualizar a posição e rotação visual das rodas
        UpdateWheelMeshes();
    }
    
    private void UpdateWheelMeshes()
    {
        foreach (WheelData wheel in wheels)
        {
            if (wheel.collider == null || wheel.wheelMesh == null) continue;
            
            Vector3 position;
            Quaternion rotation;
            wheel.collider.GetWorldPose(out position, out rotation);
            
            wheel.wheelMesh.position = position;
            wheel.wheelMesh.rotation = rotation;
        }
    }
    
    // Ferramenta de depuração para visualizar curvas de fricção (opcional)
    public void DebugFrictionCurves()
    {
        if (wheels.Length == 0 || wheels[0].collider == null) return;
        
        WheelFrictionCurve forward = wheels[0].collider.forwardFriction;
        WheelFrictionCurve sideways = wheels[0].collider.sidewaysFriction;
        
        Debug.Log($"=== Curvas de Fricção ===");
        Debug.Log($"Frente: Extremum({forward.extremumSlip}, {forward.extremumValue}), " +
                 $"Asymptote({forward.asymptoteSlip}, {forward.asymptoteValue}), Stiffness:{forward.stiffness}");
        Debug.Log($"Lateral: Extremum({sideways.extremumSlip}, {sideways.extremumValue}), " +
                 $"Asymptote({sideways.asymptoteSlip}, {sideways.asymptoteValue}), Stiffness:{sideways.stiffness}");
    }
}