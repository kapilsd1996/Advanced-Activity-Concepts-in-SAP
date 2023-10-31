import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  baseUrl = 'http://127.0.0.1:5000';

  constructor(private http: HttpClient) { }

  getTables(): Observable<string[]> {
    return this.http.get<string[]>(`${this.baseUrl}/get_tables`);
  }

  processTables(tables: string[]): Observable<any> {
    return this.http.post(`${this.baseUrl}/process_tables`, { tables });
  }

  generateOCEL(tables: string[]): Observable<any> {
    return this.http.post(`${this.baseUrl}/generate_ocel`, { tables });
}
}
