
using UnityEngine;
using UnityEngine.UI;
#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(AutopilotSystem))]
public class AutopilotSystemEditor : Editor
{
    private SerializedProperty routes;
    private SerializedProperty currentRouteIndex;
    private SerializedProperty waypointRadius;
    private SerializedProperty maxSpeed;
    private SerializedProperty corneringSpeed;
    private SerializedProperty showDebugVisuals;
    
    private bool isAddingWaypoints = false;
    private AutopilotSystem autopilot;
    
    private void OnEnable()
    {
        routes = serializedObject.FindProperty("routes");
        currentRouteIndex = serializedObject.FindProperty("currentRouteIndex");
        waypointRadius = serializedObject.FindProperty("waypointRadius");
        maxSpeed = serializedObject.FindProperty("maxSpeed");
        corneringSpeed = serializedObject.FindProperty("corneringSpeed");
        showDebugVisuals = serializedObject.FindProperty("showDebugVisuals");
        
        autopilot = (AutopilotSystem)target;
    }
    
    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        
        // Desenhar componentes padrão, agrupados por seções
        EditorGUILayout.LabelField("Rotas e Waypoints", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(routes);
        
        if (routes.arraySize > 0)
        {
            // Desenhar interface para selecionar rota atual
            EditorGUILayout.Space();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.PropertyField(currentRouteIndex);
            
            // Manter o índice dentro dos limites
            if (currentRouteIndex.intValue >= routes.arraySize)
            {
                currentRouteIndex.intValue = routes.arraySize - 1;
            }
            
            if (currentRouteIndex.intValue < 0)
            {
                currentRouteIndex.intValue = 0;
            }
            
            // Botão para adicionar nova rota
            if (GUILayout.Button("Nova Rota", GUILayout.Width(100)))
            {
                autopilot.CreateNewRoute();
                serializedObject.Update();
            }
            
            EditorGUILayout.EndHorizontal();
            
            // Gerenciar waypoints da rota atual
            if (currentRouteIndex.intValue < routes.arraySize)
            {
                SerializedProperty currentRoute = routes.GetArrayElementAtIndex(currentRouteIndex.intValue);
                SerializedProperty routeName = currentRoute.FindPropertyRelative("routeName");
                SerializedProperty waypoints = currentRoute.FindPropertyRelative("waypoints");
                SerializedProperty isLooping = currentRoute.FindPropertyRelative("isLooping");
                SerializedProperty routeColor = currentRoute.FindPropertyRelative("routeColor");
                
                EditorGUILayout.PropertyField(routeName);
                EditorGUILayout.PropertyField(isLooping);
                EditorGUILayout.PropertyField(routeColor);
                
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("Waypoints", EditorStyles.boldLabel);
                
                // Botões para adição de waypoints
                EditorGUILayout.BeginHorizontal();
                
                // Botão para adicionar waypoint na posição da câmera
                if (GUILayout.Button("Adicionar na Câmera"))
                {
                    SceneView sceneView = SceneView.lastActiveSceneView;
                    if (sceneView != null)
                    {
                        Vector3 position = sceneView.camera.transform.position;
                        position.y = 0; // Ajustar altura para nível do solo
                        autopilot.AddWaypointToCurrentRoute(position);
                        serializedObject.Update();
                    }
                }
                
                // Botão para habilitar/desabilitar modo de adição de waypoints
                string buttonText = isAddingWaypoints ? "Parar de Adicionar" : "Adicionar com Cliques";
                if (GUILayout.Button(buttonText))
                {
                    isAddingWaypoints = !isAddingWaypoints;
                    if (isAddingWaypoints)
                    {
                        EditorUtility.DisplayDialog("Modo de Adição", 
                                                 "Clique na cena para adicionar waypoints. " +
                                                 "Pressione ESC para sair do modo.", "OK");
                        SceneView.duringSceneGui += OnSceneGUI;
                    }
                    else
                    {
                        SceneView.duringSceneGui -= OnSceneGUI;
                    }
                }
                
                EditorGUILayout.EndHorizontal();
                
                if (isAddingWaypoints)
                {
                    EditorGUILayout.HelpBox("Clique na cena para adicionar waypoints. Pressione ESC para sair do modo.", 
                                           MessageType.Info);
                }
                
                if (waypoints.arraySize > 0)
                {
                    EditorGUILayout.Space();
                    EditorGUILayout.LabelField("Lista de Waypoints", EditorStyles.boldLabel);
                    
                    for (int i = 0; i < waypoints.arraySize; i++)
                    {
                        EditorGUILayout.BeginHorizontal();
                        
                        SerializedProperty waypoint = waypoints.GetArrayElementAtIndex(i);
                        SerializedProperty waypointTransform = waypoint.FindPropertyRelative("transform");
                        SerializedProperty waypointMaxSpeed = waypoint.FindPropertyRelative("maxSpeed");
                        SerializedProperty waypointWidth = waypoint.FindPropertyRelative("width");
                        
                        EditorGUILayout.LabelField("Waypoint " + i, GUILayout.Width(70));
                        
                        // Mostrar propriedades do waypoint
                        EditorGUILayout.PropertyField(waypointMaxSpeed, GUIContent.none, GUILayout.Width(50));
                        EditorGUILayout.LabelField("km/h", GUILayout.Width(30));
                        
                        // Botão para remover este waypoint
                        if (GUILayout.Button("X", GUILayout.Width(25)))
                        {
                            // Remover waypoint
                            if (EditorUtility.DisplayDialog("Confirmar", 
                                                         "Remover Waypoint " + i + "?", "Sim", "Não"))
                            {
                                // Destruir o GameObject se existir
                                SerializedProperty transformProp = waypoints.GetArrayElementAtIndex(i)
                                    .FindPropertyRelative("transform");
                                if (transformProp.objectReferenceValue != null)
                                {
                                    Object.DestroyImmediate((transformProp.objectReferenceValue as Transform).gameObject);
                                }
                                
                                // Remover do array
                                waypoints.DeleteArrayElementAtIndex(i);
                                serializedObject.ApplyModifiedProperties();
                                break;
                            }
                        }
                        
                        EditorGUILayout.EndHorizontal();
                    }
                }
                else
                {
                    EditorGUILayout.HelpBox("Esta rota não tem waypoints.", MessageType.Info);
                }
                
                // Botão para limpar todos os waypoints
                if (waypoints.arraySize > 0 && GUILayout.Button("Limpar Todos os Waypoints"))
                {
                    if (EditorUtility.DisplayDialog("Confirmar", 
                                                 "Tem certeza que deseja remover todos os waypoints desta rota?", 
                                                 "Sim", "Não"))
                    {
                        // Destruir todos os GameObjects
                        for (int i = 0; i < waypoints.arraySize; i++)
                        {
                            SerializedProperty transformProp = waypoints.GetArrayElementAtIndex(i)
                                .FindPropertyRelative("transform");
                            if (transformProp.objectReferenceValue != null)
                            {
                                Object.DestroyImmediate((transformProp.objectReferenceValue as Transform).gameObject);
                            }
                        }
                        
                        // Limpar o array
                        waypoints.ClearArray();
                        serializedObject.ApplyModifiedProperties();
                    }
                }
            }
        }
        else
        {
            EditorGUILayout.HelpBox("Adicione pelo menos uma rota para configurar o piloto automático.", 
                                  MessageType.Warning);
            
            if (GUILayout.Button("Criar Rota Inicial"))
            {
                autopilot.CreateNewRoute();
                serializedObject.Update();
            }
        }
        
        // Configurações de navegação
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Configurações de Navegação", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(waypointRadius);
        EditorGUILayout.PropertyField(maxSpeed);
        EditorGUILayout.PropertyField(corneringSpeed);
        
        // Resto das propriedades
        EditorGUILayout.Space();
        EditorGUILayout.PropertyField(showDebugVisuals);
        
        // Desenhar o resto das propriedades automaticamente
        SerializedProperty prop = serializedObject.GetIterator();
        bool enterChildren = true;
        while (prop.NextVisible(enterChildren))
        {
            enterChildren = false;
            
            // Pular propriedades já mostradas
            if (prop.name == "m_Script" || prop.name == "routes" || 
                prop.name == "currentRouteIndex" || prop.name == "waypointRadius" || 
                prop.name == "maxSpeed" || prop.name == "corneringSpeed" || 
                prop.name == "showDebugVisuals")
            {
                continue;
            }
            
            EditorGUILayout.PropertyField(prop, true);
        }
        
        serializedObject.ApplyModifiedProperties();
    }
    
    private void OnSceneGUI(SceneView sceneView)
    {
        if (!isAddingWaypoints) return;
        
        Event e = Event.current;
        
        // Botão ESC para sair do modo de adição
        if (e.type == EventType.KeyDown && e.keyCode == KeyCode.Escape)
        {
            isAddingWaypoints = false;
            SceneView.duringSceneGui -= OnSceneGUI;
            e.Use();
            return;
        }
        
        // Adicionar waypoint com clique do mouse
        if (e.type == EventType.MouseDown && e.button == 0)
        {
            Ray ray = HandleUtility.GUIPointToWorldRay(e.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                Undo.RecordObject(autopilot, "Adicionar Waypoint");
                autopilot.AddWaypointToCurrentRoute(hit.point);
                e.Use();
                
                // Forçar atualização da cena
                EditorUtility.SetDirty(autopilot);
                SceneView.RepaintAll();
            }
        }
        
        // Mostrar dica no cursor
        Handles.BeginGUI();
        Rect rect = new Rect(e.mousePosition.x + 10, e.mousePosition.y + 10, 200, 40);
        GUI.Box(rect, "Clique para adicionar\nESC para sair");
        Handles.EndGUI();
    }
}

#endif