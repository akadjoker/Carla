using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

[System.Serializable]
public class SerializableWaypointRoute
{
    public string routeName;
    public bool isLooping;
    public Color routeColor;
    public List<SerializableWaypoint> waypoints = new List<SerializableWaypoint>();
}

[System.Serializable]
public class SerializableWaypoint
{
    public Vector3 position;
    public float maxSpeed;
    public float width;
    
    public SerializableWaypoint(Waypoint waypoint)
    {
        this.position = waypoint.position;
        this.maxSpeed = waypoint.maxSpeed;
        this.width = waypoint.width;
    }
}

public class WaypointRouteIO : MonoBehaviour
{
    private const string DATA_FOLDER = "Data";
    private const string FILE_EXTENSION = ".json";
    
    private string GetDataPath()
    {
        string dataPath = Path.Combine(Application.dataPath, DATA_FOLDER);
        
        // Create the directory if it doesn't exist
        if (!Directory.Exists(dataPath))
        {
            Directory.CreateDirectory(dataPath);
            Debug.Log($"Created data directory: {dataPath}");
        }
        
        return dataPath;
    }
    
    // Save a single route to a file
    public void SaveRoute(WaypointRoute route, string filename = "")
    {
        if (string.IsNullOrEmpty(filename))
        {
            filename = route.routeName;
        }
        
        // Make sure the filename is valid
        filename = SanitizeFilename(filename);
        
        // Convert route to serializable format
        SerializableWaypointRoute serializableRoute = ConvertToSerializable(route);
        
        // Convert to JSON
        string json = JsonUtility.ToJson(serializableRoute, true); // true for pretty print
        
        // Save to file
        string filePath = Path.Combine(GetDataPath(), filename + FILE_EXTENSION);
        File.WriteAllText(filePath, json);
        
        Debug.Log($"Route '{route.routeName}' saved to {filePath}");
    }
    
  
    public WaypointRoute LoadRoute(string filename)
    {
        filename = SanitizeFilename(filename);
        string filePath = Path.Combine(GetDataPath(), filename + FILE_EXTENSION);
        
        if (!File.Exists(filePath))
        {
            Debug.LogError($"Route file not found: {filePath}");
            return null;
        }
        
        try
        {
            string json = File.ReadAllText(filePath);
            SerializableWaypointRoute serializableRoute = JsonUtility.FromJson<SerializableWaypointRoute>(json);
            
            return ConvertFromSerializable(serializableRoute);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading route from {filePath}: {e.Message}");
            return null;
        }
    }
    

    public void SaveAllRoutes(List<WaypointRoute> routes)
    {
        foreach (WaypointRoute route in routes)
        {
            SaveRoute(route);
        }
        
        Debug.Log($"Saved {routes.Count} routes to {GetDataPath()}");
    }
    

    public List<WaypointRoute> LoadAllRoutes()
    {
        List<WaypointRoute> routes = new List<WaypointRoute>();
        
        string dataPath = GetDataPath();
        string[] files = Directory.GetFiles(dataPath, "*" + FILE_EXTENSION);
        
        foreach (string file in files)
        {
            try
            {
                string json = File.ReadAllText(file);
                SerializableWaypointRoute serializableRoute = JsonUtility.FromJson<SerializableWaypointRoute>(json);
                
                WaypointRoute route = ConvertFromSerializable(serializableRoute);
                routes.Add(route);
                
                Debug.Log($"Loaded route '{route.routeName}' from {file}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Error loading route from {file}: {e.Message}");
            }
        }
        
        return routes;
    }
    

    public string[] GetAvailableRouteFilenames()
    {
        string dataPath = GetDataPath();
        string[] files = Directory.GetFiles(dataPath, "*" + FILE_EXTENSION);
        
        for (int i = 0; i < files.Length; i++)
        {
            files[i] = Path.GetFileNameWithoutExtension(files[i]);
        }
        
        return files;
    }
    

    private SerializableWaypointRoute ConvertToSerializable(WaypointRoute route)
    {
        SerializableWaypointRoute serializableRoute = new SerializableWaypointRoute
        {
            routeName = route.routeName,
            isLooping = route.isLooping,
            routeColor = route.routeColor
        };
        
        foreach (Waypoint waypoint in route.waypoints)
        {
            serializableRoute.waypoints.Add(new SerializableWaypoint(waypoint));
        }
        
        return serializableRoute;
    }

    private WaypointRoute ConvertFromSerializable(SerializableWaypointRoute serializableRoute)
    {
        WaypointRoute route = new WaypointRoute
        {
            routeName = serializableRoute.routeName,
            isLooping = serializableRoute.isLooping,
            routeColor = serializableRoute.routeColor
        };
        
        foreach (SerializableWaypoint serializableWaypoint in serializableRoute.waypoints)
        {
            Waypoint waypoint = new Waypoint(serializableWaypoint.position);
            waypoint.maxSpeed = serializableWaypoint.maxSpeed;
            waypoint.width = serializableWaypoint.width;
            
            route.waypoints.Add(waypoint);
        }
        
        return route;
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