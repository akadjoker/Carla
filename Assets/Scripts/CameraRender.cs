using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.Globalization; 
using System;
using System.IO; 

public class CameraRender : MonoBehaviour
{
    public int textureWidth = 1024;
    public int textureHeight = 1024;
    public Camera renderCamera;
    public Image uiImage; // Referência para a UI Image
    public RawImage rawImage; // Alternativa usando RawImage
    private RenderTexture renderTexture;

    [Header("Configurações de Rede")]
    private const int PORT_SEND_IMAGE = 5000;    // Porta para enviar imagens
    private const int PORT_SEND_DATA = 5002;     // Porta para enviar dados do veículo
    private const int PORT_RECEIVE = 5001;       // Porta para receber comandos
    public string serverIP = "127.0.0.1";
    public float sendDataFrequency = 0.05f;      // 20 Hz para dados
    public float sendImageFrequency = 0.033f;    // ~30 FPS para imagens
    
    private UdpClient sendImageClient;
    private UdpClient sendDataClient;
    private UdpClient receiveClient;
    private Thread receiveThread;
    private bool isRunning = true;
    
    public float speed = 0;
    public float steering = 0;
    public float breaking = 0;

    [Header("Referências do Veículo")]
    public CarInputController target;
    public Text controlModeText;               // Texto UI opcional para mostrar o modo atual
    public Text speedText;                     // Texto UI opcional para mostrar a velocidade
    public Text connectionStatusText;          // Texto UI opcional para status da conexão

   

    // Variáveis para controle manual
    private float manualSpeed = 0f;
    private float manualSteering = 0f;
    
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private Vector3 initialVelocity;
    private Vector3 initialAngularVelocity;

    // Variáveis para UI
    private float displaySpeed = 0f;
    private string connectionStatus = "Aguardando conexão...";
    private int packetsSent = 0;
    private int packetsReceived = 0;
    

     [Header("Dataset Collection")]
    public bool collectingDataset = false;
    public Text datasetStatusText;
    private string datasetDirectory;
    private string datasetImagesDirectory;
    private StreamWriter datasetFile;
    private int frameCount = 0;
    private bool autoDateSet=false;

    void Start()
    {
        // Salvar estado inicial do veículo
        if (target != null)
        {
            initialPosition = target.transform.position;
            initialRotation = target.transform.rotation;
            initialVelocity = Vector3.zero;
            initialAngularVelocity = Vector3.zero;
            
            if (target.GetComponent<Rigidbody>() != null)
            {
                Rigidbody rb = target.GetComponent<Rigidbody>();
                initialVelocity = rb.velocity;
                initialAngularVelocity = rb.angularVelocity;
            }
        }

        // Verificar se temos o componente WaypointFollower
      
        SetupCamera();
        SetupNetwork();
        StartReceiving();
        
   
        
 
        StartCoroutine(SendVehicleData());
    }

    void OnEnable()
    {
    }

    void OnDisable()
    {
        if (renderCamera != null)
            renderCamera.targetTexture = null;
            
        if (renderTexture != null)
            renderTexture.Release();
    }
    
    // Método auxiliar para converter RenderTexture para Texture2D
    private Texture2D toTexture2D(RenderTexture rTex)
    {
        Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGB24, false);
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }
    

    public RenderTexture GetRenderTexture()
    {
        return renderTexture;
    }

    void SetupCamera()
    {
        renderTexture = new RenderTexture(textureWidth, textureHeight, 24);
        renderTexture.antiAliasing = 2;
        renderTexture.filterMode = FilterMode.Bilinear;
        
        if (renderCamera == null)
            renderCamera = GetComponent<Camera>();
            
        renderCamera.targetTexture = renderTexture;
        
        //  UI Image para mostrar a render texture
        if (uiImage != null)
        {
            Sprite sprite = Sprite.Create(
                toTexture2D(renderTexture),
                new Rect(0, 0, renderTexture.width, renderTexture.height),
                new Vector2(0.5f, 0.5f)
            );
            uiImage.sprite = sprite;
        }
        
        //  recomendado para render textures)
        if (rawImage != null)
        {
            rawImage.texture = renderTexture;
        }
        
        StartCoroutine(SendCameraFrames());
    }
    
    void SetupNetwork()
    {
        sendImageClient = new UdpClient();
        sendDataClient = new UdpClient();
        receiveClient = new UdpClient(PORT_RECEIVE);
        
        connectionStatus = "Servidor ativo";
    }
    
    void StartReceiving()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.Start();
    }

    IEnumerator SendCameraFrames()
    {
        Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

        while (true)
        {
            try
            {
                // Capturar frame da câmera
                RenderTexture.active = renderTexture;
                tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
                tex.Apply();

                // Converter para JPG
                byte[] bytes = tex.EncodeToJPG(75);

                // Enviar para o servidor
                sendImageClient.Send(bytes, bytes.Length, serverIP, PORT_SEND_IMAGE);
                packetsSent++;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Erro ao enviar frame: {e.Message}");
                connectionStatus = "Erro ao enviar imagem";
                
              
            }

            yield return new WaitForSeconds(sendImageFrequency);
        }
    }
    
    IEnumerator SendVehicleData()
    {
        while (true)
        {
            try
            {
                if (target != null)
                {
                    // Criar pacote de dados do veículo
                    byte[] dataPacket = CreateVehicleDataPacket();
                    
                    // Enviar para o servidor
                    sendDataClient.Send(dataPacket, dataPacket.Length, serverIP, PORT_SEND_DATA);
                    packetsSent++;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Erro ao enviar dados do veículo: {e.Message}");
                connectionStatus = "Erro ao enviar dados";
            }
            
            yield return new WaitForSeconds(sendDataFrequency);
        }
    }
    
 
    private byte[] CreateVehicleDataPacket()
    {
        // Formato: ID(1) + PosX(4) + PosY(4) + PosZ(4) + RotY(4) + Velocidade(4) + 
        //          Steering(4) + Throttle(4) + ControlMode(1) = 30 bytes
        byte[] dataPacket = new byte[30];
        
        // ID do pacote: 0x01 para dados do veículo
        dataPacket[0] = 0x01;
        
        int offset = 1;
        
        // Posição do veículo
        BitConverter.GetBytes(target.transform.position.x).CopyTo(dataPacket, offset);
        offset += 4;
        BitConverter.GetBytes(target.transform.position.y).CopyTo(dataPacket, offset);
        offset += 4;
        BitConverter.GetBytes(target.transform.position.z).CopyTo(dataPacket, offset);
        offset += 4;
        
        // Rotação Y do veículo (direção)
        BitConverter.GetBytes(target.transform.eulerAngles.y).CopyTo(dataPacket, offset);
        offset += 4;
        
        // Velocidade atual em km/h
        BitConverter.GetBytes(target.speed).CopyTo(dataPacket, offset);
        offset += 4;
        
        // Steering input atual
        BitConverter.GetBytes(target.steerInput).CopyTo(dataPacket, offset);
        offset += 4;
        
        // Throttle input atual
        BitConverter.GetBytes(target.throttleInput).CopyTo(dataPacket, offset);
        offset += 4;
        
        // Modo de controle atual
        dataPacket[offset] = (byte)0;
        
        return dataPacket;
    }   
    void ReceiveData()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, PORT_RECEIVE);

        while (isRunning)
        {
            try
            {
                byte[] receiveBytes = receiveClient.Receive(ref remoteEndPoint);
                if (receiveBytes.Length == 12 && target.IsWeb()) // 3 floats, 4 bytes cada
                {
                    speed = BitConverter.ToSingle(receiveBytes, 0);
                    steering = BitConverter.ToSingle(receiveBytes, 4);
                    breaking = BitConverter.ToSingle(receiveBytes, 8);

                
                    float steerInput = Mathf.Clamp(steering, -1.0f, 1.0f);
                    target.SetAutopilotControls(speed,   breaking  , steerInput);
                    displaySpeed = target.speed;
                
                
                    packetsReceived++;
                    connectionStatus = "Conectado";
                }
            }
            catch (SocketException e)
            {
                Debug.LogWarning($"Erro ao receber dados: {e.Message}");
                connectionStatus = "Erro ao receber comandos";
                
            
                Thread.Sleep(10);
            }
            catch (Exception e)
            {
                Debug.LogError($"Erro inesperado ao receber dados: {e.Message}");
                connectionStatus = "Erro na conexão";
            }
        }
    }
    
    void Update()
    {
      
       if (Input.GetKeyDown(KeyCode.F5))
        {
            ToggleDatasetCollection();
        }
        
        if (Input.GetKeyDown(KeyCode.F6) && collectingDataset)
        {
           autoDateSet = !autoDateSet;
        }
        if (Input.GetKeyDown(KeyCode.F7) && collectingDataset)
        {
            CaptureFrameForDataset();
        }
        if (Input.GetKey(KeyCode.F8) && collectingDataset && (steering!=0.0f))
        {
            CaptureFrameForDataset();
        }
        
        
        // Tecla para sair da aplicação
        if (Input.GetKeyDown(KeyCode.F12))
        {
            Application.Quit();
            #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
            #endif
        }
        
        if (target == null) return;
        
      
        // Vmaos alterar para ser no WaypointSystem
        if (Input.GetKeyDown(KeyCode.F4))
        {
            ResetCar();
        }
        
        // Atualizar textos na UI
        UpdateSpeedText();
        UpdateConnectionStatusText();
        UpdateDatasetStatusText();
    }

    void FixedUpdate()
    {
        if (target == null) return;

        if (collectingDataset && datasetFile != null && autoDateSet)
        {
            CaptureFrameForDataset();
            
        }

 
        // if (target.IsWeb())
        // {
        //     float steerInput = Mathf.Clamp(steering, -1.0f, 1.0f);
        //     target.SetAutopilotControls(speed,   breaking  , steerInput);
        //     displaySpeed = target.speed;
        // }
     
    }
    
    void OnDestroy()
    {
        isRunning = false;
         if (collectingDataset && datasetFile != null)
        {
            datasetFile.Flush();
            datasetFile.Close();
        }
        if (receiveThread != null)
            receiveThread.Abort();
            
        if (sendImageClient != null)
            sendImageClient.Close();
            
        if (sendDataClient != null)
            sendDataClient.Close();
            
        if (receiveClient != null)
            receiveClient.Close();
            
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
    }
    
    void ResetCar()
    {
        if (target == null) return;

        

        Rigidbody rb = target.GetComponent<Rigidbody>();

        // Restaurar posição e rotação
        //target.transform.position = initialPosition;
        //target.transform.rotation = initialRotation;
        speed = 0;
        steering = 0;
        manualSpeed = 0;
        manualSteering = 0;

        if (rb != null)
        {
            // Garantir que o rigidbody não está em modo cinemático
            rb.isKinematic = false;
            rb.velocity = initialVelocity;
            rb.angularVelocity = initialAngularVelocity;
        }

        // Garantir que o controlador do veículo está ativado
        if (target != null)
        {
            target.enabled = true;
            target.throttleInput = 0;
            target.steerInput = 0;
        }

        Debug.Log("Carro resetado para a posição inicial.");
    }
    
  
    
 
    
    // Método para atualizar o texto de velocidade na UI
    void UpdateSpeedText()
    {
        if (speedText != null)
        {
            // Mostrar velocidade com 1 casa decimal
            speedText.text = $"Velocidade: {displaySpeed:F1} km/h";
        }
    }
    
    // Método para atualizar o texto de status da conexão
    void UpdateConnectionStatusText()
    {
        if (connectionStatusText != null)
        {
            if (collectingDataset)
            {
                connectionStatusText.text = $"Status: {connectionStatus}\nPacotes Env: {packetsSent} | Rec: {packetsReceived} ";    
            } else 
            {
            connectionStatusText.text = $"Status: {connectionStatus}\nPacotes Env: {packetsSent} | Rec: {packetsReceived}";
   
            }
        }
    }
      void CaptureFrameForDataset()
    {
        if (!collectingDataset || datasetFile == null)
        {
            Debug.LogWarning("Não é possível capturar frame: coleta não está ativa");
            return;
        }
        
        try
        {
            // Converter o RenderTexture atual para Texture2D
            Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
            RenderTexture.active = renderTexture;
            tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            tex.Apply();
            
            // Gerar nome de arquivo único com timestamp
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            string filename = $"frame_{timestamp}.jpg";
            string imagePath = $"{datasetImagesDirectory}/{filename}";
            
            // Salvar a imagem
            byte[] bytes = tex.EncodeToJPG(90);
            File.WriteAllBytes(imagePath, bytes);
            
            // Registrar apenas o steering no CSV
            if (target != null)
            {
                string steeringValue = target.steerInput.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
                datasetFile.WriteLine($"images/{filename},{steeringValue}");
                //datasetFile.WriteLine($"images/{filename},{target.steerInput:F6}");
                datasetFile.Flush();
            }
            
            frameCount++;
            Debug.Log($"Frame capturado: {imagePath} | Total: {frameCount}");
            
         
        }
        catch (Exception e)
        {
            Debug.LogError($"Erro ao capturar frame: {e.Message}");
        }
    }
    
     void UpdateDatasetStatusText()
    {
        if (datasetStatusText != null)
        {
            if (collectingDataset)
            {
                if (autoDateSet)
                    datasetStatusText.text = $"Coleta Frames: {frameCount} Auto";
                else 
                    datasetStatusText.text = $"Coleta Frames: {frameCount}";
                datasetStatusText.color = Color.green;
            }
            else
            {
                datasetStatusText.text = "";
                datasetStatusText.color = Color.red;
            }
        }
    }
     public void ToggleDatasetCollection()
    {
        if (!collectingDataset)
        {
            // Criar diretório base se não existir
            //string baseDir = Application.persistentDataPath + "/dataset";
            string baseDir = Path.Combine(Application.dataPath, "../dataset");
            if (!Directory.Exists(baseDir))
            {
                Directory.CreateDirectory(baseDir);
            }
            
            // Criar nova sessão com timestamp
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            datasetDirectory = $"{baseDir}/session_{timestamp}";
            datasetImagesDirectory = $"{datasetDirectory}/images";
            
            Directory.CreateDirectory(datasetDirectory);
            Directory.CreateDirectory(datasetImagesDirectory);
            
            // Abrir arquivo CSV apenas com image_path e steering
            datasetFile = new StreamWriter($"{datasetDirectory}/steering_data.csv");
            datasetFile.WriteLine("image_path,steering");
            datasetFile.Flush();
            
            frameCount = 0;
            collectingDataset = true;
            Debug.Log($"Iniciando coleta de dados para treino: {datasetDirectory}");
        }
        else
        {
            // Parar coleta e fechar arquivo
            if (datasetFile != null)
            {
                datasetFile.Flush();
                datasetFile.Close();
                datasetFile = null;
            }
            
            collectingDataset = false;
            Debug.Log($"Coleta de dados finalizada. Total de frames: {frameCount}");
        }
    }
}
