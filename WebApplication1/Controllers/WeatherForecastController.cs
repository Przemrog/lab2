using Microsoft.AspNetCore.Mvc;
using WebApplication1.serwisy;

namespace WebApplication1.Controllers
{
    [ApiController]
    [Route("[controller]/[action]")]
    public class WeatherForecastController : ControllerBase
    {
        private static readonly string[] Summaries = new[]
        {
        "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
    };

        private readonly ILogger<WeatherForecastController> _logger;

        public WeatherForecastController(ILogger<WeatherForecastController> logger)
        {
            _logger = logger;
        }

        [HttpGet(Name = "GetWeatherForecast")]
        public int GetValue()
        {
            var calculateInstance = new Calculate(23);
            calculateInstance.Add(1000);
            calculateInstance.Decrease(990); //powinno wyjsc 33
            return calculateInstance.GetValue();
        }

    }
}