using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(WheelSetup))]
public class WheelSetupEditor : Editor
{
    public override void OnInspectorGUI()
    {
        WheelSetup wheelSetup = (WheelSetup)target;
        
        DrawDefaultInspector();
        
        EditorGUILayout.Space();
        
        if (GUILayout.Button("Aplicar Configurações"))
        {
            wheelSetup.ConfigureWheels();
            EditorUtility.SetDirty(target);
        }
        
        if (GUILayout.Button("Visualizar Curvas de Fricção"))
        {
            wheelSetup.DebugFrictionCurves();
        }
        
        EditorGUILayout.HelpBox(
            "Dicas para configuração:\n" +
            "- Suspensão: valores mais altos de spring tornam o carro mais rígido\n" +
            "- Fricção: valores mais altos aumentam a aderência\n" +
            "- ExtremumSlip: controla quando o pneu começa a derrapar\n" +
            "- Stiffness: rigidez do pneu (1.0 = normal, <1.0 = escorregadio, >1.0 = mais aderente)",
            MessageType.Info);
    }
}
#endif