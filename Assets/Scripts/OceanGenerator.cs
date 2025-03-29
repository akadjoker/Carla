

using UnityEngine;

public class OceanGenerator : MonoBehaviour
{
    public int size = 100;        // Tamanho do oceano
    public float scale = 1.0f;    // Escala do oceano
    public Material oceanMaterial; // Material com o shader do oceano
    
    void Start()
    {
        // Cria um plano para representar o oceano
        MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = gameObject.AddComponent<MeshRenderer>();
        
        meshRenderer.material = oceanMaterial;
        
        // Cria a malha do oceano
        Mesh mesh = new Mesh();
        meshFilter.mesh = mesh;
        
        // Cria os vértices para o oceano
        Vector3[] vertices = new Vector3[(size + 1) * (size + 1)];
        Vector2[] uv = new Vector2[(size + 1) * (size + 1)];
        
        for (int z = 0; z <= size; z++)
        {
            for (int x = 0; x <= size; x++)
            {
                vertices[z * (size + 1) + x] = new Vector3(x * scale, 0, z * scale);
                uv[z * (size + 1) + x] = new Vector2((float)x / size, (float)z / size);
            }
        }
        
        // Cria os triângulos para o oceano
        int[] triangles = new int[size * size * 6];
        int triangleIndex = 0;
        
        for (int z = 0; z < size; z++)
        {
            for (int x = 0; x < size; x++)
            {
                int vertexIndex = z * (size + 1) + x;
                
                triangles[triangleIndex++] = vertexIndex;
                triangles[triangleIndex++] = vertexIndex + size + 1;
                triangles[triangleIndex++] = vertexIndex + 1;
                
                triangles[triangleIndex++] = vertexIndex + 1;
                triangles[triangleIndex++] = vertexIndex + size + 1;
                triangles[triangleIndex++] = vertexIndex + size + 2;
            }
        }
        
        // Atribui os vértices e triângulos à malha
        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = triangles;
        
        mesh.RecalculateNormals();
    }
}
