using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

[System.Serializable]
public class WaypointRoute {
    public string routeName = "Nova Rota";
    public List<Waypoint> waypoints = new List<Waypoint>();
    public bool isLooping = true;
    public Color routeColor = Color.white;
}

[System.Serializable]
public class Waypoint
{
    public Transform transform;
    public float maxSpeed = 30f; // Velocidade máxima em km/h
    public float width = 5f;     // Largura da pista neste ponto
    
    public Waypoint(Vector3 position)
    {
        GameObject waypointObj = new GameObject("Waypoint");
        transform = waypointObj.transform;
        transform.position = position;
    }
    
    public Waypoint(Transform existingTransform)
    {
        transform = existingTransform;
    }
    
    public Vector3 position
    {
        get { return transform != null ? transform.position : Vector3.zero; }
        set { if (transform != null) transform.position = value; }
    }
}

public class AutopilotSystem : MonoBehaviour
{
    [Header("Rotas e Waypoints")]
    public List<WaypointRoute> routes = new List<WaypointRoute>();
    public int currentRouteIndex = 0;
    public int currentWaypointIndex = 0;
    private Waypoint currentWaypoint;
    private Waypoint nextWaypoint;


      [Header("Route Saving/Loading")]
    public WaypointRouteIO routeIO;
    public KeyCode saveRouteKey = KeyCode.F10;
    public KeyCode loadRouteKey = KeyCode.F11;
    public Dropdown routeSelectionDropdown;
    
    [Header("Configurações de Navegação")]
    public float waypointRadius = 5.0f;     // Raio para considerar waypoint alcançado
    public float lookAheadDistance = 10.0f;  // Distância para olhar adiante da curva
    public float maxSpeed = 50.0f;           // Velocidade máxima em km/h
    public float corneringSpeed = 30.0f;     // Velocidade reduzida para curvas
    public float brakeDistance = 20.0f;      // Distância para começar a frear
    public float accelerationFactor = 1.0f;  // Fator de aceleração
    public float steeringFactor = 1.0f;      // Fator de direção
    public float obstacleDetectionDistance = 20.0f; // Distância para detectar obstáculos
    
    [Header("Componentes")]
    public CarInputController carController;
    public Rigidbody rb;
    
    [Header("Estado do Piloto Automático")]
    public bool autopilotEnabled = false;
    public KeyCode toggleAutopilotKey = KeyCode.P;
    public bool showDebugVisuals = true;
    
    [Header("UI")]
    public Text autopilotStatusText;

    public Color activeColor = Color.green;
    public Color inactiveColor = Color.red;
    
    [Header("Sensors")]
    public Transform frontSensorPosition;
    public LayerMask obstacleLayerMask;
    
    [Header("Visual")]
    public Color waypointColor = Color.yellow;
    public Color connectionColor = Color.green;
    public float gizmoSize = 0.5f;
    
    // Variáveis privadas
    private float targetSpeed = 0f;
    private float currentThrottle = 0f;
    private float currentBrake = 0f;
    private float currentSteer = 0f;
    private bool obstacleDetected = false;
    private Transform lastWaypointHit;

    public float distanceToWaypoint=0.0f;
    private Transform carTrasform;
    
    private void Start()
    {
         
        if (carController == null)
        {
            Debug.LogError("AutopilotSystem requer um componente CarInputController no mesmo GameObject.");
            enabled = false;
            return;
        }

        carTrasform = carController.transform;
        
        if (rb == null)
        {
            Debug.LogError("AutopilotSystem requer um componente Rigidbody no mesmo GameObject.");
            enabled = false;
            return;
        }
        
        // Se não houver sensor frontal definido, criar um
        if (frontSensorPosition == null)
        {
            GameObject sensorObj = new GameObject("FrontSensor");
            sensorObj.transform.parent = transform;
            sensorObj.transform.localPosition = new Vector3(0, 0.5f, 2.0f);
            frontSensorPosition = sensorObj.transform;
        }
        InitializeRouteIO();
        UpdateWaypointReferences();
        UpdateAutopilotUI();
    }
    
    private void OnApplicationQuit()
    {
        SaveAllRoutes();
    }
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.F1))
        {
            autopilotEnabled = false;
            carController.SetAutopilot(autopilotEnabled);
            Debug.Log("Autopilot desligado");
            carController.SetManual();
        }
        if (Input.GetKeyDown(KeyCode.F2))
        {
            autopilotEnabled = false;
            carController.SetAutopilot(autopilotEnabled);
            Debug.Log("Autopilot desligado");
            carController.SetWeb();
        }
        if (Input.GetKeyDown(KeyCode.F4))
        {
            ResetCar();
        }

        // Alternar piloto automático com tecla
       // if (Input.GetKeyDown(toggleAutopilotKey))
        if (Input.GetKeyDown(KeyCode.F3))
        {
        
            autopilotEnabled = !autopilotEnabled;
            carController.SetAutopilot(autopilotEnabled);
                
       
            
            // Resetar controles ao desligar
            if (!autopilotEnabled)
            {
                currentThrottle = 0f;
                currentBrake = 0f;
                currentSteer = 0f;
            }
        }
        
        UpdateAutopilotUI();
        // Trocar para próxima rota com Tab
        if (Input.GetKeyDown(KeyCode.Tab) && routes.Count > 1)
        {
            currentRouteIndex = (currentRouteIndex + 1) % routes.Count;
            currentWaypointIndex = 0;
            UpdateWaypointReferences();
        }
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            SceneManager.LoadScene(0);
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            SceneManager.LoadScene(1);
        }
    }
    
    private void FixedUpdate()
    {
        if (!autopilotEnabled) return;
        
        // Atualizar referências de waypoints se necessário
        if (currentWaypoint == null || currentWaypoint.transform == null)
        {
            UpdateWaypointReferences();
            if (currentWaypoint == null || currentWaypoint.transform == null) 
            {
                autopilotEnabled = false;

                Debug.Log("Nenhum waypoint disponível para o piloto automático.");    
                return; // Sem waypoints disponíveis
            }
        }
        
        // Verificar se chegou ao waypoint atual
         distanceToWaypoint = Vector3.Distance(carTrasform.position, currentWaypoint.position);
        if (distanceToWaypoint < waypointRadius)
        {
            // Avançar para o próximo waypoint
            currentWaypointIndex++;
            
            // Verificar se chegou ao fim da rota
            WaypointRoute currentRoute = routes[currentRouteIndex];
            if (currentWaypointIndex >= currentRoute.waypoints.Count)
            {
                if (currentRoute.isLooping)
                {
                    currentWaypointIndex = 0;
                }
                else
                {
                    
                    autopilotEnabled = false;
                    UpdateAutopilotUI();
                    return;
                }
            }
            
            UpdateWaypointReferences();
        }
        
        // Verificar obstáculos à frente
        //DetectObstacles();
        
        // Calcular direção para o waypoint
        CalculateNavigation();
        

        //Debug.Log($"Direção: {direction}, Ângulo: {angle}, Steer: {steer}");

        // Enviar comandos para o controlador do carro
        if (carController != null)
        {
             carController.SetAutopilotControls(currentThrottle, currentBrake, currentSteer);
           
        } else
        {
            Debug.LogError("AutopilotSystem requer um componente CarInputController no mesmo GameObject.");
        }
    }
    
    private void UpdateWaypointReferences()
    {
        if (routes.Count == 0 || currentRouteIndex >= routes.Count) 
        {
            Debug.LogError("Nenhuma rota disponível para o piloto automático.");
            return;
        }
        
        WaypointRoute currentRoute = routes[currentRouteIndex];
        if (currentRoute.waypoints == null || currentRoute.waypoints.Count == 0) 
        {
            Debug.LogError("Nenhum waypoint disponível para a rota atual.");
            return;
        }
        
        // Garantir que o índice esteja dentro dos limites
        currentWaypointIndex = Mathf.Clamp(currentWaypointIndex, 0, currentRoute.waypoints.Count - 1);
        
        // Atualizar waypoint atual
        currentWaypoint = currentRoute.waypoints[currentWaypointIndex];
        
        // Atualizar próximo waypoint para planejamento de curvas
        int nextIndex = (currentWaypointIndex + 1) % currentRoute.waypoints.Count;
        nextWaypoint = currentRoute.isLooping || nextIndex != 0 ? currentRoute.waypoints[nextIndex] : currentWaypoint;
    }
    
    private void CalculateNavigation()
    {
        if (currentWaypoint == null || currentWaypoint.transform == null) 
        {
            Debug.LogError("Nenhum waypoint disponível para o piloto automático.");
            return;
        }

 

        // Calcular vetor para o waypoint atual
        Vector3 directionToWaypoint = currentWaypoint.position - carTrasform.position;
        directionToWaypoint.y = 0; // Ignorar diferença de altura
        
        // Calcular ângulo relativo
        float relativeAngle = Vector3.SignedAngle(carTrasform.forward, directionToWaypoint, Vector3.up);
        
        // Normalizar para range de -1 a 1 para direção
        currentSteer = Mathf.Clamp(relativeAngle / 45f, -1f, 1f) * steeringFactor;
        
        // Calcular distância até o waypoint
        float distanceToWaypoint = directionToWaypoint.magnitude;
        
        // Calcular velocidade desejada baseada na curvatura da trajetória
        float angleToNextWaypoint = 0f;
        if (nextWaypoint != null && nextWaypoint.transform != null && nextWaypoint.transform != currentWaypoint.transform)
        {
            Vector3 currentToNext = nextWaypoint.position - currentWaypoint.position;
            angleToNextWaypoint = Vector3.Angle(directionToWaypoint, currentToNext);
        }
        
        // Determinar velocidade alvo
        if (obstacleDetected)
        {
            targetSpeed = 0f; // Parar se houver obstáculo
        }
        else
        {
            // Considerar maxSpeed definido no waypoint
            float waypointMaxSpeed = currentWaypoint.maxSpeed;
            float globalMaxSpeed = this.maxSpeed;
            float effectiveMaxSpeed = Mathf.Min(waypointMaxSpeed, globalMaxSpeed);
            
            // Velocidade baseada na curva (quanto maior o ângulo, menor a velocidade)
            float curveSpeedFactor = Mathf.Clamp01(1f - (angleToNextWaypoint / 180f));
            float desiredSpeed = Mathf.Lerp(corneringSpeed, effectiveMaxSpeed, curveSpeedFactor);
            
            // Reduzir velocidade se estiver se aproximando do waypoint
            float approachFactor = Mathf.Clamp01(distanceToWaypoint / brakeDistance);
            if (distanceToWaypoint < brakeDistance)
            {
                desiredSpeed *= approachFactor;
            }
            
            // Suavizar mudanças de velocidade alvo
            targetSpeed = Mathf.Lerp(targetSpeed, desiredSpeed, Time.fixedDeltaTime * 2f);
        }
        
        // Converter velocidade em throttle/brake
        float currentSpeed = rb.velocity.magnitude * 3.6f; // em km/h
        
        if (currentSpeed < targetSpeed)
        {
            // Acelerar
            currentThrottle = Mathf.Clamp01((targetSpeed - currentSpeed) / 20f) * accelerationFactor;
            currentBrake = 0f;
        }
        else
        {
            // Trava
            currentThrottle = 0f;
            currentBrake = Mathf.Clamp01((currentSpeed - targetSpeed) / 20f);
        }

        const float maxAngle = 60f;
        
        // Intensificar o break em ângulos acentuados
        if (Mathf.Abs(relativeAngle) > maxAngle)
        {
           // currentThrottle *= Mathf.Clamp01(1f - (Mathf.Abs(relativeAngle) - maxAngle) / maxAngle);
            //currentBrake = Mathf.Max(currentBrake, (Mathf.Abs(relativeAngle) - maxAngle) / maxAngle * 0.5f);
        }

      //  Debug.Log($"Target Speed: {targetSpeed:0} km/h | Current Speed: {currentSpeed:0} km/h | Throttle: {currentThrottle:0.00} | Brake: {currentBrake:0.00} | Steer: {currentSteer:0.00}");

    }

    private void CalculateCrazyNavigation()
{
    if (currentWaypoint == null || currentWaypoint.transform == null) return;

    // Vetor direção para o waypoint
    Vector3 direction = currentWaypoint.position - carTrasform.position;
    direction.y = 0; // Ignorar diferença de altura


    // Ângulo para o waypoint
    float angle = Vector3.SignedAngle(carTrasform.forward, direction, Vector3.up);

    // Calcular steering (direção) com zona morta para evitar oscilações
    float steer = 0f;
    if (Mathf.Abs(angle) > 5f) // Zona morta de 5 graus
    {
        steer = Mathf.Clamp(angle / 45f, -1f, 1f);
    }
    
    // Calcular throttle baseado na distância e ângulo
    float throttle = 0.2f; // Valor máximo para fazer o carro andar
    float brake = 0f;
    
    // Reduzir throttle em curvas acentuadas para não derrapar
    if (Mathf.Abs(angle) > 30f)
    {
        throttle = Mathf.Lerp(1.0f, 0.3f, (Mathf.Abs(angle) - 30f) / 60f);
    }
    
    // Aplicar travao se estiver muito perto do waypoint e em ângulo mau
    if (distanceToWaypoint < waypointRadius * 2 && Mathf.Abs(angle) > 90f)
    {
        throttle = 0.1f;
        brake = 0.5f;
    }
    
    // Suavização da direção para evitar oscilações
    currentSteer = Mathf.Lerp(currentSteer, steer, Time.fixedDeltaTime * 5f);
    
    // Se estiver muito próximo do waypoint, reduzir velocidade
    if (distanceToWaypoint < waypointRadius * 1.5f)
    {
        throttle = Mathf.Min(throttle, 0.5f);
    }

    // Atualizar valores de controle
    currentThrottle = throttle;
    currentBrake = brake;

    
    // Aplicar ao controlador
    if (carController != null)
    {
        carController.SetAutopilotControls(currentThrottle, currentBrake, currentSteer);
    }
    
    
  //  Debug.Log($"Distância: {distanceToWaypoint:0.0}m, Ângulo: {angle:0.0}°, Throttle: {currentThrottle:0.00}, Brake: {currentBrake:0.00}, Steer: {currentSteer:0.00}");
}


    private void CalculateSimplesNavigation()
{
    if (currentWaypoint == null || currentWaypoint.transform == null) return;

    // Vetor direção para o waypoint
    Vector3 direction = currentWaypoint.position - carTrasform.position;
    direction.y = 0; // Ignorar diferença de altura
    // Aqui usamos a mesma distância calculada no FixedUpdate
    // Não recalculamos distanceToWaypoint aqui

    // Ângulo para o waypoint
    float angle = Vector3.SignedAngle(carTrasform.forward, direction, Vector3.up);

    // Calcular steering (direção)
    float steer = Mathf.Clamp(angle / 45f, -1f, 1f);
    
    // Calcular throttle baseado na distância e ângulo
    float throttle = 1.0f; // Valor máximo para fazer o carro andar
    float brake = 0f;
    
    // Reduzir throttle em curvas acentuadas para não derrapar
    if (Mathf.Abs(angle) > 30f)
    {
        throttle = Mathf.Lerp(1.0f, 0.3f, (Mathf.Abs(angle) - 30f) / 60f);
    }
    
    // Aplicar freio se estiver muito perto do waypoint e em ângulo ruim
    if (distanceToWaypoint < waypointRadius * 2 && Mathf.Abs(angle) > 90f)
    {
        throttle = 0.1f;
        brake = 0.5f;
    }
    
    // Se estiver muito próximo do waypoint, reduzir velocidade
    if (distanceToWaypoint < waypointRadius * 1.5f)
    {
        throttle = Mathf.Min(throttle, 0.5f);
    }

    // Atualizar valores de controle
    currentThrottle = throttle;
    currentBrake = brake;
    currentSteer = steer;
    
    // Aplicar ao controlador
    if (carController != null)
    {
        carController.SetAutopilotControls(currentThrottle, currentBrake, currentSteer);
    }
    
    // Debug para verificar valores
   // Debug.Log($"Distância: {distanceToWaypoint:0.0}m, Ângulo: {angle:0.0}°, Throttle: {currentThrottle:0.00}, Brake: {currentBrake:0.00}, Steer: {currentSteer:0.00}");
}
    private void DetectObstacles()
    {
        obstacleDetected = false;
        
        if (frontSensorPosition == null) return;
        
        // Lançar raio à frente do veículo
        RaycastHit hit;
        if (Physics.Raycast(frontSensorPosition.position, frontSensorPosition.forward, 
                           out hit, obstacleDetectionDistance, obstacleLayerMask))
        {
            obstacleDetected = true;
            
            // Ignorar se o hit for um waypoint
            if (currentWaypoint != null && currentWaypoint.transform == hit.transform ||
                nextWaypoint != null && nextWaypoint.transform == hit.transform)
            {
                obstacleDetected = false;
            }
            
            // Evitar detectar repetidamente o mesmo waypoint
            if (hit.transform == lastWaypointHit)
            {
                obstacleDetected = false;
            }
            else if (currentWaypoint != null && hit.transform == currentWaypoint.transform || 
                    nextWaypoint != null && hit.transform == nextWaypoint.transform)
            {
                lastWaypointHit = hit.transform;
            }
            
            if (obstacleDetected && showDebugVisuals)
            {
                Debug.DrawLine(frontSensorPosition.position, hit.point, Color.red);
                Debug.Log("Obstáculo detectado: " + hit.transform.name);
            }
        }
        
        if (showDebugVisuals && !obstacleDetected)
        {
            Debug.DrawRay(frontSensorPosition.position, 
                         frontSensorPosition.forward * obstacleDetectionDistance, Color.green);
        }
    }
    

    private void UpdateAutopilotUI()
    {
        if (autopilotStatusText != null)
        {
            if (carController.IsWeb())
            {
                autopilotStatusText.text = "Server Pilot";
                autopilotStatusText.color = Color.green;
            } else if (carController.IsManual()) 
            {
                autopilotStatusText.text = "Piloto Manual";
                autopilotStatusText.color = Color.red;
            } else if (carController.IsAutopilot())
            {
                autopilotStatusText.text =  "Piloto Automático";
                autopilotStatusText.color = Color.yellow;
            }
        }
        
    
    }
    
    public void AddWaypointToCurrentRoute(Vector3 position)
    {
        if (routes.Count == 0)
        {
            // Criar nova rota se não houver nenhuma
            WaypointRoute newRoute = new WaypointRoute();
            routes.Add(newRoute);
            currentRouteIndex = 0;
        }
        
        WaypointRoute currentRoute = routes[currentRouteIndex];
        
        // Criar objeto para o waypoint
        GameObject waypointObj = new GameObject("Waypoint_" + currentRoute.waypoints.Count);
        waypointObj.transform.parent = transform;
        waypointObj.transform.position = position;
        
        // Adicionar ao array de waypoints
        Waypoint newWaypoint = new Waypoint(waypointObj.transform);
        currentRoute.waypoints.Add(newWaypoint);
        
        // Atualizar referências
        UpdateWaypointReferences();
    }
    

    public int GetClosestWaypointIndex(Vector3 position, int routeIndex)
    {
        if (routeIndex < 0 || routeIndex >= routes.Count) return -1;
        
        WaypointRoute route = routes[routeIndex];
        if (route.waypoints.Count == 0) return -1;
        
        int closestIndex = 0;
        float closestDistance = float.MaxValue;
        
        for (int i = 0; i < route.waypoints.Count; i++)
        {
            if (route.waypoints[i].transform == null) continue;
            
            float distance = Vector3.Distance(position, route.waypoints[i].position);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestIndex = i;
            }
        }
        
        return closestIndex;
    }
    
    public void CreateNewRoute()
    {
        WaypointRoute newRoute = new WaypointRoute();
        newRoute.routeName = "Rota " + routes.Count;
        routes.Add(newRoute);
        currentRouteIndex = routes.Count - 1;
    }
    
    // Função para visualizar as rotas no Editor
    private void OnDrawGizmos()
    {
        if (!showDebugVisuals) return;
        
        // Desenhar todas as rotas
        for (int r = 0; r < routes.Count; r++)
        {
            WaypointRoute route = routes[r];
            if (route.waypoints == null || route.waypoints.Count == 0) continue;
            
            // Definir cor da rota
            Gizmos.color = route.routeColor;
            
            // Desenhar linha entre waypoints
            for (int i = 0; i < route.waypoints.Count; i++)
            {
                if (route.waypoints[i].transform == null) continue;
                
                Vector3 position = route.waypoints[i].position;
                
                // Destacar rota atual
                if (r == currentRouteIndex)
                {
                    // Destacar waypoint atual
                    if (i == currentWaypointIndex)
                    {
                        Gizmos.DrawSphere(position, waypointRadius * 0.5f);
                    }
                    else
                    {
                        Gizmos.DrawSphere(position, gizmoSize);
                    }
                    
                    // Mostrar velocidade máxima
                    #if UNITY_EDITOR
                    UnityEditor.Handles.Label(position + Vector3.up * 2, 
                        $"{i}: {route.waypoints[i].maxSpeed} km/h");
                    #endif
                }
                else
                {
                    Gizmos.DrawSphere(position, gizmoSize * 0.5f);
                }
                
                // Desenhar linhas conectando os waypoints
                if (i + 1 < route.waypoints.Count && route.waypoints[i + 1].transform != null)
                {
                    Vector3 nextPosition = route.waypoints[i + 1].position;
                    Gizmos.DrawLine(position, nextPosition);
                    
                    // Desenhar seta direcional
                    Vector3 direction = (nextPosition - position).normalized;
                    Vector3 arrowPos = position + direction * Vector3.Distance(position, nextPosition) * 0.5f;
                    float arrowSize = gizmoSize * 2;
                    
                    Gizmos.DrawRay(arrowPos, Quaternion.Euler(0, 30, 0) * direction * arrowSize);
                    Gizmos.DrawRay(arrowPos, Quaternion.Euler(0, -30, 0) * direction * arrowSize);
                }
                else if (route.isLooping && route.waypoints[0].transform != null)
                {
                    // Conectar o último ao primeiro se for loop
                    Vector3 nextPosition = route.waypoints[0].position;
                    Gizmos.DrawLine(position, nextPosition);
                }
            }
        }
        
        // Desenhar informações sobre o waypoint atual
        if (autopilotEnabled && currentWaypoint != null && currentWaypoint.transform != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(carTrasform.position, currentWaypoint.position);
            
            // Desenhar círculo de alcance do waypoint
            DrawWireDisc(currentWaypoint.position, Vector3.up, waypointRadius, 32);
        }
    }
    

    private void DrawWireDisc(Vector3 position, Vector3 normal, float radius, int segments)
    {
        Vector3 from = Vector3.up;
        if (normal == from || normal == -from)
        {
            from = Vector3.forward;
        }
        
        Vector3 right = Vector3.Cross(normal, from).normalized * radius;
        Vector3 forward = Vector3.Cross(normal, right).normalized * radius;
        
        for (int i = 0; i < segments; i++)
        {
            float angle1 = (float)i / segments * 2 * Mathf.PI;
            float angle2 = (float)(i + 1) / segments * 2 * Mathf.PI;
            
            Vector3 pos1 = position + right * Mathf.Cos(angle1) + forward * Mathf.Sin(angle1);
            Vector3 pos2 = position + right * Mathf.Cos(angle2) + forward * Mathf.Sin(angle2);
            
            Gizmos.DrawLine(pos1, pos2);
        }
    }
     private void InitializeRouteIO()
    {
   
        if (routeIO == null)
        {
            routeIO = GetComponent<WaypointRouteIO>();
            if (routeIO == null)
            {
                routeIO = gameObject.AddComponent<WaypointRouteIO>();
            }
        }
        
     
        UpdateRouteDropdown();
    }
    

    private void HandleRouteSaveLoad()
    {
      
        if (Input.GetKeyDown(saveRouteKey))
        {
            SaveCurrentRoute();
        }
        
        
        if (Input.GetKeyDown(loadRouteKey))
        {
            LoadSavedRoutes();
        }
    }
    

    public void SaveCurrentRoute()
    {
        if (routes.Count == 0 || currentRouteIndex >= routes.Count)
        {
            Debug.LogWarning("No route available to save.");
            return;
        }
        
        WaypointRoute currentRoute = routes[currentRouteIndex];
        routeIO.SaveRoute(currentRoute);
        
        UpdateRouteDropdown();
    }
    
 
    public void SaveAllRoutes()
    {
        if (routes.Count == 0)
        {
            Debug.LogWarning("No routes available to save.");
            return;
        }
        
        routeIO.SaveAllRoutes(routes);
        
  
        UpdateRouteDropdown();
    }
    

    public void LoadSavedRoutes()
    {
        List<WaypointRoute> loadedRoutes = routeIO.LoadAllRoutes();
        
        if (loadedRoutes.Count == 0)
        {
            Debug.LogWarning("No routes found to load.");
            return;
        }
        
  
        routes = loadedRoutes;
        currentRouteIndex = 0;
        currentWaypointIndex = 0;
        
        UpdateWaypointReferences();
        UpdateRouteDropdown();
        
        Debug.Log($"Loaded {routes.Count} routes.");
    }
    

    public void LoadRoute(string filename)
    {
        WaypointRoute loadedRoute = routeIO.LoadRoute(filename);
        
        if (loadedRoute == null)
        {
            Debug.LogWarning($"Failed to load route: {filename}");
            return;
        }
        

        bool routeExists = false;
        int routeIndex = 0;
        
        for (int i = 0; i < routes.Count; i++)
        {
            if (routes[i].routeName == loadedRoute.routeName)
            {
                routes[i] = loadedRoute;
                routeExists = true;
                routeIndex = i;
                break;
            }
        }
        
        if (!routeExists)
        {
            routes.Add(loadedRoute);
            routeIndex = routes.Count - 1;
        }
        
     
        currentRouteIndex = routeIndex;
        currentWaypointIndex = 0;
        
        UpdateWaypointReferences();
        UpdateRouteDropdown();
        
        Debug.Log($"Loaded route: {loadedRoute.routeName}");
    }
    

    private void UpdateRouteDropdown()
    {
        if (routeSelectionDropdown == null) return;
        
  
        routeSelectionDropdown.ClearOptions();
        
    
        List<string> options = new List<string>();
        
     
        foreach (WaypointRoute route in routes)
        {
            options.Add(route.routeName);
        }
        
  
        string[] savedRoutes = routeIO.GetAvailableRouteFilenames();
        
   
        foreach (string routeName in savedRoutes)
        {
            if (!options.Contains(routeName))
            {
                options.Add(routeName);
            }
        }
        

        routeSelectionDropdown.AddOptions(options);
        
 
        if (routes.Count > 0 && currentRouteIndex < routes.Count)
        {
            int index = options.IndexOf(routes[currentRouteIndex].routeName);
            if (index >= 0)
            {
                routeSelectionDropdown.value = index;
            }
        }
        
 
        routeSelectionDropdown.onValueChanged.RemoveAllListeners();
        routeSelectionDropdown.onValueChanged.AddListener(OnRouteDropdownChanged);
    }
    

    private void OnRouteDropdownChanged(int index)
    {
        if (routeSelectionDropdown == null) return;
        
        string selectedRouteName = routeSelectionDropdown.options[index].text;
        
     
        for (int i = 0; i < routes.Count; i++)
        {
            if (routes[i].routeName == selectedRouteName)
            {
                currentRouteIndex = i;
                currentWaypointIndex = 0;
                UpdateWaypointReferences();
                return;
            }
        }
        

        LoadRoute(selectedRouteName);
    }
    void ResetCar()
    {
        if (carTrasform == null) return;
        WaypointRoute currentRoute = routes[currentRouteIndex];
        if (currentRoute.waypoints.Count<=1) return;


        
        
 
        Vector3 initialPosition = currentRoute.waypoints[0].position;
        Vector3 targetPosition = currentRoute.waypoints[1].position;
        Quaternion targetRotation = Quaternion.LookRotation(targetPosition - initialPosition);
        
        

        Rigidbody rb = carController.GetComponent<Rigidbody>();

        // Restaurar posição e rotação
        carTrasform.position = initialPosition;
        carTrasform.rotation = targetRotation;


        if (rb != null)
        {
        
            rb.isKinematic = false;
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

     
         
            carController.enabled = true;
            carController.throttleInput = 0;
            carController.steerInput = 0;
        

        Debug.Log("Carro resetado para a posição inicial.");
    }
}
