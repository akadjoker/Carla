using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class VehicleConfigSaver : MonoBehaviour
{
    [Header("Referências")]
    public CarInputController carInputController;
    public WheelSetup wheelSetup;

    
    [Header("Configurações de Salvamento")]
    public string configFolderName = "VehicleConfigs";
    public string defaultConfigName = "DefaultConfig";
    

    [System.Serializable]
    public class SerializableCarConfig
    {
        // Configurações de Controle
        public float motorForce;
        public float brakeForce;
        public float maxSteerAngle;
        public float steeringSpeed;
        
        // Configurações Avançadas
        public bool useAutomaticGearbox;
        public float[] gearRatios;
        public float reverseGearRatio;
        public float downshiftRPM;
        public float upshiftRPM;
        public float maxRPM;
        public float idleRPM;
        
        // Auxiliares de Direção
        public bool useAntiRollBars;
        public float antiRollForce;
        public bool useTractionControl;
        public float tractionSlipLimit;
    }
    
    [System.Serializable]
    public class SerializableWheelConfig
    {
        // Configurações de Suspensão
        public float suspensionDistance;
        public float suspensionSpring;
        public float suspensionDamper;
        public float suspensionTargetPosition;
        
        // Configurações de Fricção
        public float forwardFriction;
        public float sidewaysFriction;
        public float forwardExtremumSlip;
        public float sidewaysExtremumSlip;
        public float forwardAsymptoteSlip;
        public float sidewaysAsymptoteSlip;
        public float forwardStiffness;
        public float sidewaysStiffness;
        
        // Configurações Avançadas
        public float wheelMass;
        public float wheelRadius;
        public float wheelDampingRate;
    }
    
    [System.Serializable]
    public class CompleteVehicleConfig
    {
        public SerializableCarConfig carConfig;
        public SerializableWheelConfig wheelConfig;
       
    }
    
    private void Start()
    {

        string configPath = Path.Combine(Application.dataPath, configFolderName);
        if (!Directory.Exists(configPath))
        {
            Directory.CreateDirectory(configPath);
            Debug.Log($"Diretório de configurações criado: {configPath}");
        }
    }

    private void OnApplicationQuit()
    {
        SaveCompleteVehicleConfig();
     
    }
    

    public void SaveCompleteVehicleConfig(string configName = "")
    {
        if (string.IsNullOrEmpty(configName))
        {
            configName = defaultConfigName;
        }
        
        // Sanitizar nome do arquivo
        configName = SanitizeFilename(configName);
        
        CompleteVehicleConfig config = new CompleteVehicleConfig
        {
            carConfig = ExtractCarConfig(),
            wheelConfig = ExtractWheelConfig(),

        };
        
        // Converter para JSON
        string json = JsonUtility.ToJson(config, true); // true para formatação bonita
        
        // Salvar em arquivo
        string configPath = Path.Combine(Application.dataPath, configFolderName);
        string filePath = Path.Combine(configPath, configName + ".json");
        File.WriteAllText(filePath, json);
        
        Debug.Log($"Configuração completa salva em: {filePath}");
    }
    
    // Carregar configurações completas do veículo e rotas
    public void LoadCompleteVehicleConfig(string configName = "")
    {
        if (string.IsNullOrEmpty(configName))
        {
            configName = defaultConfigName;
        }
        
        // Sanitizar nome do arquivo
        configName = SanitizeFilename(configName);
        
        string configPath = Path.Combine(Application.dataPath, configFolderName);
        string filePath = Path.Combine(configPath, configName + ".json");
        
        if (!File.Exists(filePath))
        {
            Debug.LogWarning($"Configuração não encontrada: {filePath}");
            return;
        }
        
        try
        {
            string json = File.ReadAllText(filePath);
            CompleteVehicleConfig config = JsonUtility.FromJson<CompleteVehicleConfig>(json);
            
      
            ApplyCarConfig(config.carConfig);
            ApplyWheelConfig(config.wheelConfig);
  
            
            Debug.Log($"Configuração carregada com sucesso: {filePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Erro ao carregar configuração: {e.Message}");
        }
    }
    
    
    
    // Extrair configuração do CarInputController
    private SerializableCarConfig ExtractCarConfig()
    {
        if (carInputController == null)
        {
            Debug.LogWarning("CarInputController não encontrado.");
            return new SerializableCarConfig();
        }
        
        SerializableCarConfig config = new SerializableCarConfig
        {
            motorForce = carInputController.motorForce,
            brakeForce = carInputController.brakeForce,
            maxSteerAngle = carInputController.maxSteerAngle,
            steeringSpeed = carInputController.steeringSpeed,
            
            useAutomaticGearbox = carInputController.useAutomaticGearbox,
            gearRatios = carInputController.gearRatios,
            reverseGearRatio = carInputController.reverseGearRatio,
            downshiftRPM = carInputController.downshiftRPM,
            upshiftRPM = carInputController.upshiftRPM,
            maxRPM = carInputController.maxRPM,
            idleRPM = carInputController.idleRPM,
            
            useAntiRollBars = carInputController.useAntiRollBars,
            antiRollForce = carInputController.antiRollForce,
            useTractionControl = carInputController.useTractionControl,
            tractionSlipLimit = carInputController.tractionSlipLimit
        };
        
        return config;
    }
    
    // Extrair configuração do WheelSetup
    private SerializableWheelConfig ExtractWheelConfig()
    {
        if (wheelSetup == null)
        {
            Debug.LogWarning("WheelSetup não encontrado.");
            return new SerializableWheelConfig();
        }
        
        SerializableWheelConfig config = new SerializableWheelConfig
        {
            suspensionDistance = wheelSetup.suspensionDistance,
            suspensionSpring = wheelSetup.suspensionSpring,
            suspensionDamper = wheelSetup.suspensionDamper,
            suspensionTargetPosition = wheelSetup.suspensionTargetPosition,
            
            forwardFriction = wheelSetup.forwardFriction,
            sidewaysFriction = wheelSetup.sidewaysFriction,
            forwardExtremumSlip = wheelSetup.forwardExtremumSlip,
            sidewaysExtremumSlip = wheelSetup.sidewaysExtremumSlip,
            forwardAsymptoteSlip = wheelSetup.forwardAsymptoteSlip,
            sidewaysAsymptoteSlip = wheelSetup.sidewaysAsymptoteSlip,
            forwardStiffness = wheelSetup.forwardStiffness,
            sidewaysStiffness = wheelSetup.sidewaysStiffness,
            
            wheelMass = wheelSetup.wheelMass,
            wheelRadius = wheelSetup.wheelRadius,
            wheelDampingRate = wheelSetup.wheelDampingRate
        };
        
        return config;
    }
    
    
    // Aplicar configuração ao CarInputController
    private void ApplyCarConfig(SerializableCarConfig config)
    {
        if (carInputController == null || config == null)
        {
            return;
        }
        
        carInputController.motorForce = config.motorForce;
        carInputController.brakeForce = config.brakeForce;
        carInputController.maxSteerAngle = config.maxSteerAngle;
        carInputController.steeringSpeed = config.steeringSpeed;
        
        carInputController.useAutomaticGearbox = config.useAutomaticGearbox;
        carInputController.gearRatios = config.gearRatios;
        carInputController.reverseGearRatio = config.reverseGearRatio;
        carInputController.downshiftRPM = config.downshiftRPM;
        carInputController.upshiftRPM = config.upshiftRPM;
        carInputController.maxRPM = config.maxRPM;
        carInputController.idleRPM = config.idleRPM;
        
        carInputController.useAntiRollBars = config.useAntiRollBars;
        carInputController.antiRollForce = config.antiRollForce;
        carInputController.useTractionControl = config.useTractionControl;
        carInputController.tractionSlipLimit = config.tractionSlipLimit;
    }
    
    // Aplicar configuração ao WheelSetup
    private void ApplyWheelConfig(SerializableWheelConfig config)
    {
        if (wheelSetup == null || config == null)
        {
            return;
        }
        
        wheelSetup.suspensionDistance = config.suspensionDistance;
        wheelSetup.suspensionSpring = config.suspensionSpring;
        wheelSetup.suspensionDamper = config.suspensionDamper;
        wheelSetup.suspensionTargetPosition = config.suspensionTargetPosition;
        
        wheelSetup.forwardFriction = config.forwardFriction;
        wheelSetup.sidewaysFriction = config.sidewaysFriction;
        wheelSetup.forwardExtremumSlip = config.forwardExtremumSlip;
        wheelSetup.sidewaysExtremumSlip = config.sidewaysExtremumSlip;
        wheelSetup.forwardAsymptoteSlip = config.forwardAsymptoteSlip;
        wheelSetup.sidewaysAsymptoteSlip = config.sidewaysAsymptoteSlip;
        wheelSetup.forwardStiffness = config.forwardStiffness;
        wheelSetup.sidewaysStiffness = config.sidewaysStiffness;
        
        wheelSetup.wheelMass = config.wheelMass;
        wheelSetup.wheelRadius = config.wheelRadius;
        wheelSetup.wheelDampingRate = config.wheelDampingRate;
    }
        

    private string SanitizeFilename(string filename)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
        {
            filename = filename.Replace(c, '_');
        }
        
        return filename;
    }
}